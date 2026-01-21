"""Model factories and sklearn wrappers (extracted, best-effort)."""

from __future__ import annotations

# These blocks reference notebook globals such as SEED / USE_GPU.
# In the refactor, move such values into a config object.
def _make_lgbm(**kw):
    kw.setdefault("random_state", SEED)
    # kw.setdefault("deterministic", True)
    # kw.setdefault("force_row_wise", True) 
    kw.setdefault("feature_fraction_seed", SEED)
    kw.setdefault("data_random_seed", SEED)
    kw.setdefault("device", 'gpu' if USE_GPU else 'cpu')
    return lightgbm.LGBMClassifier(**kw)


def _make_xgb(**kw):
    kw.setdefault("random_state", SEED)
    kw.setdefault("tree_method", "gpu_hist" if USE_GPU else "hist")
    # kw.setdefault("deterministic_histogram", True)
    return XGBClassifier(**kw)


def _make_cb(**kw):
    kw.setdefault("random_seed", SEED)
    if USE_GPU:
        kw.setdefault("task_type", "GPU")
        kw.setdefault("devices", "0")
    else:
        kw.setdefault("task_type", "CPU")

    return CatBoostClassifier(**kw)


# ================= StratifiedSubsetClassifier =================


class StratifiedSubsetClassifierWEval(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 estimator,
                 n_samples=None,
                 random_state: int = 42,
                 valid_size: float = 0.10,
                 val_cap_ratio: float = 0.25,
                 es_rounds: "int|str" = "auto",
                 es_metric: str = "auto"):
        self.estimator = estimator
        self.n_samples = (int(n_samples) if (n_samples is not None) else None)
        self.random_state = random_state
        self.valid_size = float(valid_size)
        self.val_cap_ratio = float(val_cap_ratio)
        self.es_rounds = es_rounds
        self.es_metric = es_metric
 
    # -------------------------- API --------------------------
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        n_total = len(y); assert n_total == len(X)

        tr_idx, va_idx = self._compute_train_val_indices(y, n_total)
        Xtr = X.iloc[tr_idx]; ytr = y[tr_idx]

        Xtr = Xtr.to_numpy(np.float32, copy=False)

        Xva = yva = None
        if va_idx is not None and len(va_idx) > 0:
            Xva = X.iloc[va_idx].to_numpy(np.float32, copy=False); yva = y[va_idx]

        # Compute pos_rate on VALIDATION (what ES monitors)
        pos_rate = None
        if yva is not None and len(yva) > 0:
            pos_rate = float(np.mean(yva == 1))

        # Decide metric & patience
        metric = self._choose_metric(pos_rate)
        patience = self._choose_patience(pos_rate)

        # Apply imbalance knobs per library
        if self._is_xgb(self.estimator):
            # scale_pos_weight = n_neg / n_pos on TRAIN
            n_pos = max(1, int((ytr == 1).sum()))
            n_neg = max(1, len(ytr) - n_pos)
            self.estimator.set_params(scale_pos_weight=(n_neg / n_pos))
            self.estimator.set_params(eval_metric=metric)

        elif self._is_catboost(self.estimator):
            # GPU-safe auto balancing
            try: self.estimator.set_params(auto_class_weights="Balanced")
            except Exception: pass
            try: self.estimator.set_params(eval_metric=metric)
            except Exception: pass

        # Fit with ES if we have any validation (single-class OK with Logloss)
        has_valid = (Xva is not None and len(yva) > 0)
        if has_valid and self._is_xgb(self.estimator):
            import xgboost as xgb
            self.estimator.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(
                    rounds=int(patience),
                    metric_name=metric,
                    data_name="validation_0",
                    save_best=True
                )]
            )
        elif has_valid and self._is_catboost(self.estimator):
            from catboost import Pool
            self.estimator.set_params(
                use_best_model=True,
                od_type="Iter",
                od_wait=int(patience),
                custom_metric=["PRAUC:type=Classic;hints=skip_train~true"],
            )
            self.estimator.fit(
                Xtr, ytr,
                eval_set=Pool(Xva, yva),
                verbose=False,
                metric_period=50
            )
        else:
            # Fall back: train on train split without ES
            self.estimator.fit(Xtr, ytr)

        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        self._tr_idx_ = tr_idx; self._va_idx_ = va_idx; self._pos_rate_ = pos_rate
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.estimator.predict_proba(X)

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(X)

    # -------------------------- helpers --------------------------
    def _compute_train_val_indices(self, y: np.ndarray, n_total: int):
        rng = np.random.default_rng(self.random_state)
        n_classes = np.unique(y).size

        def full_data_split():
            if self.valid_size <= 0 or n_classes < 2:
                idx = rng.permutation(n_total); return idx, None
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.valid_size, random_state=self.random_state)
            tr, va = next(sss.split(np.zeros(n_total, dtype=np.int8), y))
            return tr, va

        if self.n_samples is None or self.n_samples >= n_total:
            return full_data_split()

        # Use n_samples for train; build val from remainder (capped)
        sss_tr = StratifiedShuffleSplit(n_splits=1, train_size=self.n_samples, random_state=self.random_state)
        tr_idx, rest_idx = next(sss_tr.split(np.zeros(n_total, dtype=np.int8), y))
        remaining = len(rest_idx)

        min_val_needed = int(np.ceil(self.n_samples * max(self.valid_size, 0.0)))
        val_cap = max(min_val_needed, int(round(self.val_cap_ratio * self.n_samples)))
        want_val = min(remaining, val_cap)

        y_rest = y[rest_idx]
        if remaining < min_val_needed or np.unique(y_rest).size < 2 or self.valid_size <= 0:
            return full_data_split()

        sss_val = StratifiedShuffleSplit(n_splits=1, train_size=want_val, random_state=self.random_state)
        try:
            va_sel, _ = next(sss_val.split(np.zeros(remaining, dtype=np.int8), y_rest))
        except ValueError:
            return full_data_split()

        va_idx = rest_idx[va_sel]
        return tr_idx, va_idx

    def _choose_metric(self, pos_rate=0.01) -> str:
        if self.es_metric != "auto":
            return self.es_metric
        if pos_rate is None or pos_rate == 0.0 or pos_rate == 1.0:
            return "logloss" if self._is_xgb(self.estimator) else "Logloss"
        return "aucpr" if self._is_xgb(self.estimator) else "PRAUC:type=Classic"

    def _choose_patience(self, pos_rate: Optional[float]) -> int:
        if isinstance(self.es_rounds, int):
            return self.es_rounds
        try:
            n_estimators = (int(self.estimator.get_params().get("n_estimators", 200))
                            if self._is_xgb(self.estimator)
                            else int(self.estimator.get_params().get("iterations", 500)))
        except Exception:
            n_estimators = 200
        base = max(30, int(round(0.20 * (n_estimators or 200))))
        if pos_rate is None:
            return base
        if pos_rate < 0.005:   # <0.5%
            return int(round(base * 1.75))
        if pos_rate < 0.02:    # <2%
            return int(round(base * 1.40))
        return base

    @staticmethod
    def _is_xgb(est):
        name = est.__class__.__name__.lower(); mod = getattr(est, "__module__", "")
        return "xgb" in name or "xgboost" in mod or hasattr(est, "get_xgb_params")

    @staticmethod
    def _is_catboost(est):
        name = est.__class__.__name__.lower(); mod = getattr(est, "__module__", "")
        return "catboost" in name or "catboost" in mod or hasattr(est, "get_all_params")


class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples, random_state=SEED):
        self.estimator = estimator
        self.n_samples = n_samples and int(n_samples)
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        n_total = len(y)

        if self.n_samples is None or self.n_samples >= n_total:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(n_total)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=self.n_samples, random_state=self.random_state
            )
            idx, _ = next(sss.split(np.zeros(n_total, dtype=np.int8), y))

        Xn = X.iloc[idx]
        Xn = Xn.to_numpy(np.float32, copy=False)
        yn = y[idx]

        self.estimator.fit(Xn, yn)
        self.classes_ = getattr(self.estimator, "classes_", np.array([0, 1]))
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return self.estimator.predict(X)


# ==================== SCORING FUNCTIONS ====================


class HostVisibleError(Exception):
    pass

