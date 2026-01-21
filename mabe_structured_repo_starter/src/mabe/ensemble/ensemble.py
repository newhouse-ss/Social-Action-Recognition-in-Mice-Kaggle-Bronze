"""Ensembling + thresholding utilities (extracted, best-effort)."""

from __future__ import annotations
import numpy as np
import pandas as pd

def _find_lgbm_step(pipe):
    try:
        if "stratifiedsubsetclassifier__estimator" in pipe.get_params():
            est = pipe.get_params()["stratifiedsubsetclassifier__estimator"]
            if isinstance(est, lightgbm.LGBMClassifier):
                return "stratifiedsubsetclassifier"
        if "stratifiedsubsetclassifierweval__estimator" in pipe.get_params():
            est = pipe.get_params()["stratifiedsubsetclassifierweval__estimator"]
            if isinstance(est, lightgbm.LGBMClassifier):
                return "stratifiedsubsetclassifierweval"
    except Exception as e:
        print(e)
    return None


def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta, n_samples=1_500_000):
    models = []
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=225, learning_rate=0.07, min_child_samples=40,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1, gpu_use_dp=USE_GPU
        ), n_samples)
    ))
    models.append(make_pipeline(
        StratifiedSubsetClassifier(_make_lgbm(
            n_estimators=150, learning_rate=0.1, min_child_samples=20,
            num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1, gpu_use_dp=USE_GPU
        ), (n_samples and int(n_samples/1.25)))
    ))
    # models.append(make_pipeline(
    #     StratifiedSubsetClassifier(_make_lgbm(
    #         n_estimators=100, learning_rate=0.05, min_child_samples=30,
    #         num_leaves=127, max_depth=10, subsample=0.75, verbose=-1, gpu_use_dp=USE_GPU,
    #     ), (n_samples and int(n_samples/1.66)))
    # ))

    xgb0 = _make_xgb(
        n_estimators=180, learning_rate=0.08, max_depth=6,
        min_child_weight=8 if USE_GPU else 5, gamma=1.0 if USE_GPU else 0.,
        subsample=0.8, colsample_bytree=0.8, single_precision_histogram=USE_GPU,
        verbosity=0
    )
    models.append(make_pipeline(StratifiedSubsetClassifier(xgb0, n_samples and int(n_samples/1.2))))

    # cb_est = _make_cb(iterations=120, learning_rate=0.1, depth=6,
    #                   verbose=False, allow_writing_files=False)
    # models.append(make_pipeline(StratifiedSubsetClassifier(cb_est, n_samples)))

    model_names = ['lgbm_225', 'lgbm_150', 'lgbm_100', 'xgb_180', 'cat_120']

    if USE_GPU:
        xgb1 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=2000, learning_rate=0.05, grow_policy="lossguide",
            max_leaves=255, max_depth=0, min_child_weight=10, gamma=0.0,
            subsample=0.90, colsample_bytree=1.00, colsample_bylevel=0.85,
            reg_alpha=0.0, reg_lambda=1.0, max_bin=256,
            single_precision_histogram=True, verbosity=0
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(xgb1, n_samples and int(n_samples/2.),
                                            random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                                            es_rounds="auto", es_metric="auto")
        ))
        xgb2 = XGBClassifier(
            random_state=SEED, booster="gbtree", tree_method="gpu_hist",
            n_estimators=1400, learning_rate=0.06, max_depth=7,
            min_child_weight=12, subsample=0.70, colsample_bytree=0.80,
            reg_alpha=0.0, reg_lambda=1.5, max_bin=256,
            single_precision_histogram=True, verbosity=0
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(xgb2, n_samples and int(n_samples/1.5),
                                            random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                                            es_rounds="auto", es_metric="auto")
        ))

        cb1 = CatBoostClassifier(
            random_seed=SEED, task_type="GPU", devices="0",
            iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=6.0,
            bootstrap_type="Bayesian", bagging_temperature=0.5,
            random_strength=0.5, loss_function="Logloss",
            eval_metric="PRAUC:type=Classic", auto_class_weights="Balanced",
            border_count=64, verbose=False, allow_writing_files=False
        )
        models.append(make_pipeline(
            StratifiedSubsetClassifierWEval(cb1, n_samples and int(n_samples/2.0),
                                            random_state=SEED, valid_size=0.10, val_cap_ratio=0.25,
                                            es_rounds="auto", es_metric="auto")
        ))
        model_names.extend(['xgb1', 'xgb2', 'cat_bay'])

    model_list = []
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        meta_masked = meta.iloc[action_mask]
  
        trained = []
        for model_idx, m in enumerate(models):
            m_clone = clone(m)
            try:
                t0 = perf_counter()
                m_clone.fit(X_tr[action_mask], y_action)
                dt = perf_counter() - t0
                print(f"trained model {model_names[model_idx]} | {switch_tr} | action={action} | {dt:.1f}s", flush=True)
            except Exception:
                step = _find_lgbm_step(m_clone)
                if step is None:
                    continue
                try:
                    m_clone.set_params(**{f"{step}__estimator__device": "cpu"})
                    t0 = perf_counter()
                    m_clone.fit(X_tr[action_mask], y_action)
                    dt = perf_counter() - t0
                    print(f"trained (CPU fallback) {model_names[model_idx]} | {switch_tr} | action={action} | {dt:.1f}s", flush=True)
                except Exception as e2:
                    print(e2)
                    continue
            trained.append(m_clone)

        if trained:
            model_list.append((action, trained))

    del X_tr; gc.collect()

    # ---- TEST INFERENCE ----
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )
    fps_lookup = (test_subset[['video_id','frames_per_second']]
                    .drop_duplicates('video_id')
                    .set_index('video_id')['frames_per_second'].to_dict())

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)

            del data_te

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    probs = []
                    for mi, mdl in enumerate(trained):
                        probs.append(mdl.predict_proba(X_te)[:, 1])
                    pred[action] = np.mean(probs, axis=0)

            del X_te; gc.collect()

            if pred.shape[1] != 0:
                submission_list.append(predict_multiclass_adaptive(pred, meta_te))
        except Exception as e:
            print(e)
            try: del data_te
            except: pass
            gc.collect()


def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]

    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop:
                mask[i] = False
            else:
                last_stop = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    # if len(s_list) > 0:
    #     submission = pd.concat([
    #         submission,
    #         pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
    #     ])

    submission = submission.reset_index(drop=True)
    return submission

# ==================== MAIN LOOP ====================

submission_list = []

for section in range(len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

        _fps_lookup = (
            train_subset[['video_id', 'frames_per_second']]
            .drop_duplicates('video_id')
            .set_index('video_id')['frames_per_second']
            .to_dict()
        )

        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []

        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            else:
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)

        if len(single_list) > 0:
            single_feats_parts = []
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(Xi)

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
 
            single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
            single_meta  = pd.concat(single_meta_list,  axis=0, ignore_index=True)

            del single_list, single_label_list, single_meta_list, single_feats_parts
            gc.collect()

            print(f"  Single: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'single', X_tr, single_label, single_meta)

            del X_tr, single_label, single_meta
            gc.collect()

        if len(pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(Xi)

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)

            
            pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
            pair_meta  = pd.concat(pair_meta_list,  axis=0, ignore_index=True)

            del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
            gc.collect()

            print(f"  Pair: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta)

            del X_tr, pair_label, pair_meta
            gc.collect()

    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')

    gc.collect()
    print()

if len(submission_list) > 0:
    submission = pd.concat(submission_list, ignore_index=True)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission_robust = robustify(submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nSubmission created: {len(submission_robust)} predictions")


def predict_multiclass_adaptive(pred, meta, action_thresholds=defaultdict(lambda: 0.24)):
    """Adaptive thresholding per action + temporal smoothing"""
    # Apply temporal smoothing
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()
    
    ama = np.argmax(pred_smoothed, axis=1)
    
    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = action_thresholds.get(action, 0.24)
        threshold_mask |= (action_mask & (max_probs >= threshold))
    
    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    
    # Filter out very short events (likely noise)
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part

# ==================== ADVANCED FEATURE ENGINEERING (FPS-AWARE) ====================

