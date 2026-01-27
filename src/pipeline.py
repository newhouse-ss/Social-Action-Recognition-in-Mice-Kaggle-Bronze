from __future__ import annotations

import gc
import json
import os
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from .config import DROP_BODY_PARTS, SEED, USE_GPU, resolve_data_dir, set_seed
from .data import generate_mouse_data, load_train_test
from .features import _fps_from_meta, transform_pair, transform_single
from .models import (
    StratifiedSubsetClassifier,
    StratifiedSubsetClassifierWEval,
    _find_lgbm_step,
    _make_cb,
    _make_lgbm,
    _make_xgb,
)
from .postprocess import predict_multiclass_adaptive, robustify


def _build_feature_matrix(
    data_list,
    label_list,
    meta_list,
    body_parts_tracked,
    fps_lookup,
    mode: str,
):
    feats_parts = []
    for data_i, meta_i in zip(data_list, meta_list):
        fps_i = _fps_from_meta(meta_i, fps_lookup, default_fps=30.0)
        if mode == "single":
            Xi = transform_single(data_i, body_parts_tracked, fps_i)
        else:
            Xi = transform_pair(data_i, body_parts_tracked, fps_i)
        feats_parts.append(Xi.astype(np.float32))

    X_tr = pd.concat(feats_parts, axis=0, ignore_index=True)
    label = pd.concat(label_list, axis=0, ignore_index=True)
    meta = pd.concat(meta_list, axis=0, ignore_index=True)
    return X_tr, label, meta


def submit_ensemble(
    body_parts_tracked_str: str,
    switch_tr: str,
    X_tr: pd.DataFrame,
    label: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    data_dir: str,
    n_samples: Optional[int] = 1_500_000,
    use_gpu: Optional[bool] = None,
    seed: int = SEED,
    drop_body_parts: Optional[list[str]] = None,
    verbose: bool = False,
) -> list[pd.DataFrame]:
    drop_body_parts = DROP_BODY_PARTS if drop_body_parts is None else drop_body_parts
    use_gpu = USE_GPU if use_gpu is None else use_gpu
    submission_parts: list[pd.DataFrame] = []

    models = []
    models.append(
        make_pipeline(
            StratifiedSubsetClassifier(
                _make_lgbm(
                    n_estimators=225,
                    learning_rate=0.07,
                    min_child_samples=40,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    verbose=-1,
                    gpu_use_dp=use_gpu,
                    use_gpu=use_gpu,
                ),
                n_samples,
            )
        )
    )
    models.append(
        make_pipeline(
            StratifiedSubsetClassifier(
                _make_lgbm(
                    n_estimators=150,
                    learning_rate=0.1,
                    min_child_samples=20,
                    num_leaves=63,
                    max_depth=8,
                    subsample=0.7,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    verbose=-1,
                    gpu_use_dp=use_gpu,
                    use_gpu=use_gpu,
                ),
                (n_samples and int(n_samples / 1.25)),
            )
        )
    )

    xgb0 = _make_xgb(
        n_estimators=180,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=8 if use_gpu else 5,
        gamma=1.0 if use_gpu else 0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        single_precision_histogram=use_gpu,
        verbosity=0,
        use_gpu=use_gpu,
    )
    models.append(make_pipeline(StratifiedSubsetClassifier(xgb0, n_samples and int(n_samples / 1.2))))

    model_names = ["lgbm_225", "lgbm_150", "lgbm_100", "xgb_180", "cat_120"]

    if use_gpu:
        xgb1 = XGBClassifier(
            random_state=seed,
            booster="gbtree",
            tree_method="gpu_hist",
            n_estimators=2000,
            learning_rate=0.05,
            grow_policy="lossguide",
            max_leaves=255,
            max_depth=0,
            min_child_weight=10,
            gamma=0.0,
            subsample=0.90,
            colsample_bytree=1.00,
            colsample_bylevel=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            max_bin=256,
            single_precision_histogram=True,
            verbosity=0,
        )
        models.append(
            make_pipeline(
                StratifiedSubsetClassifierWEval(
                    xgb1,
                    n_samples and int(n_samples / 2.0),
                    random_state=seed,
                    valid_size=0.10,
                    val_cap_ratio=0.25,
                    es_rounds="auto",
                    es_metric="auto",
                )
            )
        )
        xgb2 = XGBClassifier(
            random_state=seed,
            booster="gbtree",
            tree_method="gpu_hist",
            n_estimators=1400,
            learning_rate=0.06,
            max_depth=7,
            min_child_weight=12,
            subsample=0.70,
            colsample_bytree=0.80,
            reg_alpha=0.0,
            reg_lambda=1.5,
            max_bin=256,
            single_precision_histogram=True,
            verbosity=0,
        )
        models.append(
            make_pipeline(
                StratifiedSubsetClassifierWEval(
                    xgb2,
                    n_samples and int(n_samples / 1.5),
                    random_state=seed,
                    valid_size=0.10,
                    val_cap_ratio=0.25,
                    es_rounds="auto",
                    es_metric="auto",
                )
            )
        )

        cb1 = CatBoostClassifier(
            random_seed=seed,
            task_type="GPU",
            devices="0",
            iterations=4000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=6.0,
            bootstrap_type="Bayesian",
            bagging_temperature=0.5,
            random_strength=0.5,
            loss_function="Logloss",
            eval_metric="PRAUC:type=Classic",
            auto_class_weights="Balanced",
            border_count=64,
            verbose=False,
            allow_writing_files=False,
        )
        models.append(
            make_pipeline(
                StratifiedSubsetClassifierWEval(
                    cb1,
                    n_samples and int(n_samples / 2.0),
                    random_state=seed,
                    valid_size=0.10,
                    val_cap_ratio=0.25,
                    es_rounds="auto",
                    es_metric="auto",
                )
            )
        )
        model_names.extend(["xgb1", "xgb2", "cat_bay"])

    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)

        trained = []
        for model_idx, m in enumerate(models):
            m_clone = clone(m)
            try:
                t0 = perf_counter()
                m_clone.fit(X_tr[action_mask], y_action)
                dt = perf_counter() - t0
                if verbose:
                    print(
                        f"trained model {model_names[model_idx]} | {switch_tr} | action={action} | {dt:.1f}s",
                        flush=True,
                    )
            except Exception:
                step = _find_lgbm_step(m_clone)
                if step is None:
                    continue
                try:
                    m_clone.set_params(**{f"{step}__estimator__device": "cpu"})
                    t0 = perf_counter()
                    m_clone.fit(X_tr[action_mask], y_action)
                    dt = perf_counter() - t0
                    if verbose:
                        print(
                            f"trained (CPU fallback) {model_names[model_idx]} | {switch_tr} | action={action} | {dt:.1f}s",
                            flush=True,
                        )
                except Exception as e2:
                    if verbose:
                        print(e2)
                    continue
            trained.append(m_clone)

        if trained:
            model_list.append((action, trained))

    del X_tr
    gc.collect()

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test_df[test_df.body_parts_tracked == body_parts_tracked_str]
    test_tracking_dir = os.path.join(data_dir, "test_tracking")
    generator = generate_mouse_data(
        test_subset,
        "test",
        traintest_directory=test_tracking_dir,
        generate_single=(switch_tr == "single"),
        generate_pair=(switch_tr == "pair"),
        drop_body_parts=drop_body_parts,
        verbose=verbose,
    )
    fps_lookup = (
        test_subset[["video_id", "frames_per_second"]]
        .drop_duplicates("video_id")
        .set_index("video_id")["frames_per_second"]
        .to_dict()
    )

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            if switch_te == "single":
                X_te = transform_single(data_te, body_parts_tracked, fps_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)

            del data_te

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    probs = []
                    for mdl in trained:
                        probs.append(mdl.predict_proba(X_te)[:, 1])
                    pred[action] = np.mean(probs, axis=0)

            del X_te
            gc.collect()

            if pred.shape[1] != 0:
                submission_parts.append(predict_multiclass_adaptive(pred, meta_te, verbose=verbose))
        except Exception as e:
            if verbose:
                print(e)
            try:
                del data_te
            except Exception:
                pass
            gc.collect()

    return submission_parts


def run_pipeline(
    data_dir: Optional[str] = None,
    output_path: str = "submission.csv",
    n_samples: Optional[int] = 1_500_000,
    use_gpu: Optional[bool] = None,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    data_dir = resolve_data_dir(data_dir)
    use_gpu = USE_GPU if use_gpu is None else use_gpu
    set_seed(seed)

    train, test = load_train_test(data_dir)
    body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

    submission_parts: list[pd.DataFrame] = []
    train_tracking_dir = os.path.join(data_dir, "train_tracking")
    train_annotation_dir = os.path.join(data_dir, "train_annotation")

    for section, body_parts_tracked_str in enumerate(body_parts_tracked_list):
        try:
            body_parts_tracked = json.loads(body_parts_tracked_str)
            if verbose:
                print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
            if len(body_parts_tracked) > 5:
                body_parts_tracked = [b for b in body_parts_tracked if b not in DROP_BODY_PARTS]

            train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

            fps_lookup = (
                train_subset[["video_id", "frames_per_second"]]
                .drop_duplicates("video_id")
                .set_index("video_id")["frames_per_second"]
                .to_dict()
            )

            single_list, single_label_list, single_meta_list = [], [], []
            pair_list, pair_label_list, pair_meta_list = [], [], []

            generator = generate_mouse_data(
                train_subset,
                "train",
                traintest_directory=train_tracking_dir,
                annotation_directory=train_annotation_dir,
                drop_body_parts=DROP_BODY_PARTS,
                verbose=verbose,
            )

            for switch, data, meta, label in generator:
                if switch == "single":
                    single_list.append(data)
                    single_meta_list.append(meta)
                    single_label_list.append(label)
                else:
                    pair_list.append(data)
                    pair_meta_list.append(meta)
                    pair_label_list.append(label)

            if len(single_list) > 0:
                X_tr, single_label, single_meta = _build_feature_matrix(
                    single_list,
                    single_label_list,
                    single_meta_list,
                    body_parts_tracked,
                    fps_lookup,
                    mode="single",
                )

                if verbose:
                    print(f"  Single: {X_tr.shape}")
                submission_parts.extend(
                    submit_ensemble(
                        body_parts_tracked_str,
                        "single",
                        X_tr,
                        single_label,
                        single_meta,
                        test_df=test,
                        data_dir=data_dir,
                        n_samples=n_samples,
                        use_gpu=use_gpu,
                        seed=seed,
                        drop_body_parts=DROP_BODY_PARTS,
                        verbose=verbose,
                    )
                )

                del X_tr, single_label, single_meta
                gc.collect()

            if len(pair_list) > 0:
                X_tr, pair_label, pair_meta = _build_feature_matrix(
                    pair_list,
                    pair_label_list,
                    pair_meta_list,
                    body_parts_tracked,
                    fps_lookup,
                    mode="pair",
                )

                if verbose:
                    print(f"  Pair: {X_tr.shape}")
                submission_parts.extend(
                    submit_ensemble(
                        body_parts_tracked_str,
                        "pair",
                        X_tr,
                        pair_label,
                        pair_meta,
                        test_df=test,
                        data_dir=data_dir,
                        n_samples=n_samples,
                        use_gpu=use_gpu,
                        seed=seed,
                        drop_body_parts=DROP_BODY_PARTS,
                        verbose=verbose,
                    )
                )

                del X_tr, pair_label, pair_meta
                gc.collect()

        except Exception as e:
            if verbose:
                print(f"***Exception*** {str(e)[:100]}")

        gc.collect()
        if verbose:
            print()

    if len(submission_parts) > 0:
        submission = pd.concat(submission_parts, ignore_index=True)
    else:
        submission = pd.DataFrame(
            {
                "video_id": [438887472],
                "agent_id": ["mouse1"],
                "target_id": ["self"],
                "action": ["rear"],
                "start_frame": [278],
                "stop_frame": [500],
            }
        )

    test_tracking_dir = os.path.join(data_dir, "test_tracking")
    submission_robust = robustify(
        submission,
        test,
        "test",
        traintest_directory=test_tracking_dir,
        verbose=verbose,
    )
    submission_robust.index.name = "row_id"
    submission_robust.to_csv(output_path)
    if verbose:
        print(f"\nSubmission created: {len(submission_robust)} predictions")

    return submission_robust


__all__ = ["run_pipeline", "submit_ensemble"]
