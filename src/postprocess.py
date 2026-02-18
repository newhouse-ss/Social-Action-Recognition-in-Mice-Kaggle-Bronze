"""Post-processing: convert frame-level probabilities to action intervals.

The pipeline converts continuous per-frame probability scores into discrete,
non-overlapping event intervals suitable for submission:

  1. Threshold probabilities per action
  2. Best-label decoding (argmax among passing actions)
  3. Short-gap filling (merge nearby segments of the same action)
  4. Minimum-length filtering (discard very short events)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ALL_ACTIONS, InferConfig


# ---------------------------------------------------------------------------
# Probability → intervals
# ---------------------------------------------------------------------------


def probs_to_intervals(
    prob_df: pd.DataFrame,
    actions: List[str],
    thresholds: Optional[Dict[str, float]] = None,
    min_len: int = 3,
    max_gap: int = 2,
    default_threshold: float = 0.5,
) -> pd.DataFrame:
    """Convert frame-level probabilities to non-overlapping action intervals.

    For each (video, agent, target) group:
      1. Apply per-action thresholds to identify candidate frames.
      2. Select the best action per frame via argmax among passing actions.
      3. Fill short gaps (≤ *max_gap* frames) between same-action runs.
      4. Drop runs shorter than *min_len* frames.

    Returns:
        DataFrame with columns: ``video_id``, ``agent_id``, ``target_id``,
        ``action``, ``start_frame``, ``stop_frame``.
    """
    if thresholds is None:
        thresholds = {}
    act_cols = [f"action_{a}" for a in actions]
    thr = np.array(
        [thresholds.get(a, default_threshold) for a in actions], dtype=np.float32
    )

    out: list[dict] = []

    for (vid, ag, tg), grp in prob_df.groupby(
        ["video_id", "agent_id", "target_id"], sort=False
    ):
        g = grp.sort_values("frame")
        frames = g["frame"].to_numpy()
        P = g[act_cols].to_numpy(np.float32)

        # --- Threshold → best label ---
        pass_mask = P >= thr[None, :]
        P_masked = np.where(pass_mask, P, -np.inf)
        best_idx = np.argmax(P_masked, axis=1)
        best_val = P_masked[np.arange(len(P_masked)), best_idx]
        label = np.where(np.isfinite(best_val), best_idx, -1)

        # --- Fill short gaps ---
        if max_gap > 0:
            i = 0
            while i < len(label):
                if label[i] >= 0:
                    j = i
                    while j + 1 < len(label) and label[j + 1] == label[i]:
                        j += 1
                    k = j + 1
                    while k < len(label) and label[k] == -1:
                        k += 1
                    if (
                        k < len(label)
                        and label[k] == label[i]
                        and (k - j - 1) <= max_gap
                    ):
                        label[j + 1 : k] = label[i]
                        j = k
                    i = j + 1
                else:
                    i += 1

        # --- Convert runs to intervals ---
        def flush(s, e, idx):
            if s is None:
                return
            if frames[e] - frames[s] + 1 >= min_len:
                out.append(
                    {
                        "video_id": int(vid),
                        "agent_id": int(ag),
                        "target_id": int(tg),
                        "action": actions[idx],
                        "start_frame": int(frames[s]),
                        "stop_frame": int(frames[e] + 1),
                    }
                )

        s_run = None
        cur = -1
        for i_t, idx in enumerate(label):
            if idx != cur:
                if cur >= 0:
                    flush(s_run, i_t - 1, cur)
                s_run = i_t if idx >= 0 else None
                cur = idx
        if cur >= 0:
            flush(s_run, len(label) - 1, cur)

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Format submission
# ---------------------------------------------------------------------------


def format_submission(intervals: pd.DataFrame) -> pd.DataFrame:
    """Format interval DataFrame into the expected submission schema.

    - Converts numeric agent/target IDs to ``mouse<N>`` / ``self`` format.
    - Adds a ``row_id`` column.
    - Deduplicates and sorts.
    """
    if intervals.empty:
        return pd.DataFrame(
            columns=[
                "row_id", "video_id", "agent_id", "target_id",
                "action", "start_frame", "stop_frame",
            ]
        )

    sub = intervals.copy()
    sub = sub[sub["stop_frame"] > sub["start_frame"]]
    sub = sub[sub["action"].isin(ALL_ACTIONS)]

    # Convert IDs
    sub["agent_id"] = sub["agent_id"].apply(lambda n: f"mouse{int(n)}")
    sub["target_id"] = [
        "self" if int(ai) == int(ti) else f"mouse{int(ti)}"
        for ai, ti in zip(
            sub["agent_id"].str.replace("mouse", "").astype(int),
            intervals.loc[sub.index, "target_id"].astype(int),
        )
    ]

    cols = ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
    sub = sub[cols].drop_duplicates().reset_index(drop=True)
    sub.insert(0, "row_id", range(len(sub)))
    return sub
