from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


def predict_multiclass_adaptive(
    pred: pd.DataFrame,
    meta: pd.DataFrame,
    action_thresholds: Optional[dict] = None,
    verbose: bool = False,
):
    if action_thresholds is None:
        action_thresholds = defaultdict(lambda: 0.24)

    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()

    ama = np.argmax(pred_smoothed, axis=1)

    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = ama == i
        threshold = action_thresholds.get(action, 0.24)
        threshold_mask |= action_mask & (max_probs >= threshold)

    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)

    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False

    submission_part = pd.DataFrame(
        {
            "video_id": meta_changes["video_id"][mask].values,
            "agent_id": meta_changes["agent_id"][mask].values,
            "target_id": meta_changes["target_id"][mask].values,
            "action": pred.columns[ama_changes[mask].values],
            "start_frame": ama_changes.index[mask],
            "stop_frame": ama_changes.index[1:][mask[:-1]],
        }
    )

    stop_video_id = meta_changes["video_id"][1:][mask[:-1]].values
    stop_agent_id = meta_changes["agent_id"][1:][mask[:-1]].values
    stop_target_id = meta_changes["target_id"][1:][mask[:-1]].values

    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc("stop_frame")] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc("stop_frame")] = new_stop_frame

    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)

    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), "stop <= start"

    if verbose:
        print(f"  actions found: {len(submission_part)}")
    return submission_part


def robustify(
    submission: pd.DataFrame,
    dataset: pd.DataFrame,
    traintest: str,
    traintest_directory: Optional[str] = None,
    verbose: bool = False,
):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]

    group_list = []
    for _, group in submission.groupby(["video_id", "agent_id", "target_id"]):
        group = group.sort_values("start_frame")
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row["start_frame"] < last_stop:
                mask[i] = False
            else:
                last_stop = row["stop_frame"]
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row["lab_id"]
        video_id = row["video_id"]
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = eval(row["behaviors_labeled"])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(",") for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=["agent", "target", "action"])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(["agent", "target"]):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row["action"], batch_start, batch_stop))

    submission = submission.reset_index(drop=True)
    return submission


__all__ = ["predict_multiclass_adaptive", "robustify"]
