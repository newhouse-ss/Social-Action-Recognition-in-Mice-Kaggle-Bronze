"""Inference pipeline for the CNN-Transformer model.

Provides:
  - ``predict_video``    – sliding-window inference for one video
  - ``smooth_probs``     – temporal smoothing of frame-level probabilities
  - ``mask_probs``       – zero out probabilities for actions not annotated
                           for a given (video, agent, target)
  - ``ensemble_predict`` – average predictions across multiple model checkpoints
"""

from __future__ import annotations

import contextlib
import gc
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ALL_ACTIONS, InferConfig, SELF_ACTIONS, PAIR_ACTIONS


# ---------------------------------------------------------------------------
# Core sliding-window prediction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def predict_groups(
    model: nn.Module,
    actions: List[str],
    df: pd.DataFrame,
    feat_cols: List[str],
    window_size: int = 128,
    window_step: int = 64,
    batch_windows: int = 2048,
    device: str = "cuda",
) -> pd.DataFrame:
    """Sliding-window inference over grouped (video, agent, target) sequences.

    For each group the model predicts overlapping windows; overlapping frame
    predictions are averaged to produce a single probability per frame.

    Returns:
        DataFrame with columns ``[video_id, agent_id, target_id, frame]``
        plus ``action_<name>`` columns for every action.
    """
    id_cols = ["video_id", "frame", "agent_id", "target_id"]
    act_cols = [f"action_{a}" for a in actions]

    gobj = df.groupby(["video_id", "agent_id", "target_id"], sort=False)
    out_parts = []

    for (vid, a, t), grp in gobj:
        g = grp.sort_values("frame")
        X = g[feat_cols].to_numpy(np.float32, copy=False)
        F = len(g)

        preds = np.zeros((F, len(actions)), np.float32)
        counts = np.zeros((F, 1), np.float32)

        starts = list(range(0, F, window_step))
        wi = 0
        while wi < len(starts):
            batch_starts = starts[wi : wi + batch_windows]
            if not batch_starts:
                break

            lens = []
            maxT = 0
            for s in batch_starts:
                e = min(s + window_size, F)
                lens.append((s, e))
                maxT = max(maxT, e - s)

            batch = np.empty((len(lens), maxT, X.shape[1]), np.float32)
            for i, (s, e) in enumerate(lens):
                L = e - s
                batch[i, :L] = X[s:e]
                if L < maxT:
                    batch[i, L:maxT] = X[e - 1 : e]

            tb = torch.from_numpy(batch).to(device, non_blocking=True)
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if torch.cuda.is_available()
                else contextlib.nullcontext()
            )
            with ctx:
                out = model(tb).sigmoid().detach().cpu().numpy().astype(np.float32)

            for i, (s, e) in enumerate(lens):
                L = e - s
                preds[s:e] += out[i, :L]
                counts[s:e] += 1.0

            del batch, tb, out
            wi += batch_windows

        preds /= np.maximum(counts, 1.0)
        part = pd.DataFrame(preds, columns=act_cols)
        part["video_id"] = int(vid)
        part["agent_id"] = int(a)
        part["target_id"] = int(t)
        part["frame"] = g["frame"].to_numpy(copy=False)
        out_parts.append(part)

        del g, X, preds, counts, part, grp

    if out_parts:
        return pd.concat(out_parts, ignore_index=True, copy=False)
    return pd.DataFrame(columns=id_cols + act_cols)


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------


def smooth_probs(
    probs: pd.DataFrame,
    actions: List[str],
    win: int = 5,
) -> pd.DataFrame:
    """Apply a uniform moving-average smoothing kernel to action probabilities."""
    if win <= 1 or probs.empty:
        return probs

    probs = probs.copy()
    act_cols = [f"action_{a}" for a in actions]
    probs.sort_values(["video_id", "agent_id", "target_id", "frame"], inplace=True)
    pad = win // 2
    kernel = np.ones(win, dtype=np.float32) / win

    for _, idx in probs.groupby(
        ["video_id", "agent_id", "target_id"], sort=False
    ).groups.items():
        idx = np.asarray(idx)
        arr = probs.loc[idx, act_cols].to_numpy(np.float32, copy=False)
        for j in range(arr.shape[1]):
            v = arr[:, j]
            if v.size:
                vp = np.pad(v, (pad, pad), mode="edge")
                arr[:, j] = np.convolve(vp, kernel, mode="valid").astype(np.float32)
        probs.loc[idx, act_cols] = arr

    return probs


# ---------------------------------------------------------------------------
# Probability masking
# ---------------------------------------------------------------------------


def mask_probs(
    probs: pd.DataFrame,
    actions: List[str],
    active_map: Dict[int, Set[str]],
) -> pd.DataFrame:
    """Zero out probabilities for actions not in the active set of each video.

    Also enforces self-action / pair-action constraints:
      - Self-actions can only fire when agent == target
      - Pair-actions can only fire when agent != target
    """
    probs = probs.copy()
    act_cols = [f"action_{a}" for a in actions]

    # Self vs pair constraints
    self_cols = [f"action_{a}" for a in SELF_ACTIONS if a in actions]
    pair_cols = [f"action_{a}" for a in PAIR_ACTIONS if a in actions]
    if self_cols:
        probs.loc[probs["agent_id"] != probs["target_id"], self_cols] = 0.0
    if pair_cols:
        probs.loc[probs["agent_id"] == probs["target_id"], pair_cols] = 0.0

    # Per-video active-set masking
    if not active_map:
        return probs

    act_block = probs[act_cols].to_numpy(copy=False)
    vid = probs["video_id"].to_numpy(np.int64, copy=False)
    ag = probs["agent_id"].to_numpy(np.int64, copy=False)
    tg = probs["target_id"].to_numpy(np.int64, copy=False)
    act_pos = {a: i for i, a in enumerate(actions)}

    N = len(probs)
    allow: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
    for v, triples in active_map.items():
        d = allow.setdefault(int(v), {})
        for s in triples:
            parts = s.split(",")
            key = (int(parts[0]), int(parts[1]))
            arr = d.get(key)
            if arr is None:
                arr = np.zeros(len(actions), dtype=bool)
                d[key] = arr
            idx = act_pos.get(parts[2])
            if idx is not None:
                arr[idx] = True

    # RLE-style masking
    if N <= 1:
        starts = np.array([0], dtype=np.int64)
        ends = np.array([N], dtype=np.int64)
    else:
        change = (vid[1:] != vid[:-1]) | (ag[1:] != ag[:-1]) | (tg[1:] != tg[:-1])
        boundaries = np.flatnonzero(change) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [N]))

    for s, e in zip(starts, ends):
        v_i, a_i, t_i = int(vid[s]), int(ag[s]), int(tg[s])
        d = allow.get(v_i)
        if d is None:
            act_block[s:e, :] = 0.0
            continue
        mask = d.get((a_i, t_i))
        if mask is None:
            act_block[s:e, :] = 0.0
            continue
        if not mask.all():
            act_block[s:e, ~mask] = 0.0

    probs[act_cols] = act_block
    return probs


# ---------------------------------------------------------------------------
# Multi-model ensemble
# ---------------------------------------------------------------------------


def ensemble_predict(
    models: List[nn.Module],
    actions: List[str],
    df: pd.DataFrame,
    feat_cols: List[str],
    cfg: InferConfig,
    device: str = "cuda",
) -> pd.DataFrame:
    """Average predictions from multiple model checkpoints.

    Each model runs sliding-window inference independently; the resulting
    per-frame probabilities are averaged across all models.
    """
    all_probs = []
    for i, model in enumerate(models, 1):
        model.eval()
        model.to(device)
        probs = predict_groups(
            model,
            actions,
            df,
            feat_cols,
            window_size=cfg.window_size,
            window_step=cfg.window_step,
            batch_windows=cfg.batch_windows,
            device=device,
        )
        all_probs.append(probs)
        model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_probs) == 1:
        return all_probs[0]

    key = ["video_id", "agent_id", "target_id", "frame"]
    act_cols = [f"action_{a}" for a in actions]
    merged = all_probs[0]
    for k in range(1, len(all_probs)):
        merged = merged.merge(all_probs[k], on=key, how="inner", suffixes=(None, f"__m{k}"))

    for a in act_cols:
        like = [c for c in merged.columns if c.split("__")[0] == a]
        merged[a] = merged[like].mean(axis=1).astype(np.float32)

    return merged[key + act_cols]
