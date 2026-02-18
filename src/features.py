"""Feature engineering for the CNN-Transformer pipeline.

Converts raw tracking parquet data into per-frame pairwise feature matrices:

  1. Load & pivot tracking data → per-mouse wide DataFrames
  2. Normalize coordinates (pix_per_cm, arena centering)
  3. Fill missing values and create validity masks
  4. Add velocity features (dx/dt, dy/dt per body part)
  5. Build pairwise features: agent stream | target stream | relative features
  6. Robust scaling (median / IQR, clipped to [-5, 5])
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DROP_BODY_PARTS


# ---------------------------------------------------------------------------
# 1. Tracking I/O & pivoting
# ---------------------------------------------------------------------------


def load_tracking(path: str) -> pd.DataFrame:
    """Read a single tracking parquet file."""
    return pd.read_parquet(path)


def pivot_mouse_frame(
    track: pd.DataFrame, all_parts: List[str]
) -> Dict[int, pd.DataFrame]:
    """Pivot long-form tracking into per-mouse wide DataFrames.

    Returns:
        {mouse_id: DataFrame} where each DataFrame is indexed by
        ``video_frame`` with columns ``x_<part>``, ``y_<part>`` for every
        body part present.
    """
    per_mouse: Dict[int, pd.DataFrame] = {}
    for mid, grp in track.groupby("mouse_id", sort=False):
        pv = grp.pivot(index="video_frame", columns="bodypart", values=["x", "y"])
        pv.columns = [f"{coord}_{part}" for coord, part in pv.columns]
        pv = pv.sort_index()
        per_mouse[int(mid)] = pv
    return per_mouse


# ---------------------------------------------------------------------------
# 2. Normalize coordinates
# ---------------------------------------------------------------------------


def normalize_coords(
    df: pd.DataFrame,
    pix_per_cm: float,
    width_pix: float,
    height_pix: float,
) -> pd.DataFrame:
    """Convert pixel coordinates to centimeters, centered on the arena."""
    df = df.copy()
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]

    cx = width_pix / 2.0
    cy = height_pix / 2.0

    for c in x_cols:
        df[c] = (df[c] - cx) / pix_per_cm
    for c in y_cols:
        df[c] = (df[c] - cy) / pix_per_cm

    return df


# ---------------------------------------------------------------------------
# 3. Fill missing & create masks
# ---------------------------------------------------------------------------


def fill_and_create_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill NaN, backward-fill remainder, and add mask columns.

    For every ``x_<part>`` column, a ``m_x_<part>`` mask column is created:
    1.0 where the original value was valid, 0.0 where it was missing.
    """
    mask_cols = {}
    for c in df.columns:
        mask_cols[f"m_{c}"] = df[c].notna().astype(np.float32)

    df = df.ffill().bfill().fillna(0.0)

    for mc, vals in mask_cols.items():
        df[mc] = vals

    return df


# ---------------------------------------------------------------------------
# 4. Velocity features
# ---------------------------------------------------------------------------


def add_velocities(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Add velocity columns (vx_<part>, vy_<part>) scaled by FPS."""
    new_cols = {}
    for c in list(df.columns):
        if c.startswith("x_"):
            part = c[2:]
            new_cols[f"vx_{part}"] = df[c].diff().fillna(0.0) * fps
        elif c.startswith("y_"):
            part = c[2:]
            new_cols[f"vy_{part}"] = df[c].diff().fillna(0.0) * fps

    for k, v in new_cols.items():
        df[k] = v.astype(np.float32)

    # Also add masked velocity versions
    for c in list(df.columns):
        if c.startswith("m_x_"):
            part = c[4:]
            df[f"m_vx_{part}"] = df[c]
        elif c.startswith("m_y_"):
            part = c[4:]
            df[f"m_vy_{part}"] = df[c]

    return df


# ---------------------------------------------------------------------------
# 5. Pairwise feature construction
# ---------------------------------------------------------------------------


def _make_target_prefixed(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix target-mouse columns with ``t_`` (or ``m_t_`` for masks)."""
    keep = [
        c
        for c in df.columns
        if c.startswith(("x_", "y_", "vx_", "vy_", "m_x_", "m_y_", "m_vx_", "m_vy_"))
    ]
    mapping = {c: ("m_t_" + c[2:] if c.startswith("m_") else "t_" + c) for c in keep}
    return df[keep].rename(columns=mapping)


def build_pairwise_features(
    per_mouse: Dict[int, pd.DataFrame],
    allowed_pairs: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Build feature matrix for all (agent, target) pairs.

    Returns a DataFrame with columns:
        ``frame``, ``agent_id``, ``target_id``,
        agent features, target features (``t_`` prefix), and
        relative features (``rel_dx``, ``rel_dy``, ``rel_dist``).
    """
    if not per_mouse:
        return pd.DataFrame(columns=["frame", "agent_id", "target_id"])

    mice = sorted(per_mouse.keys())

    # Align all mice to common frame index
    common_idx = per_mouse[mice[0]].index
    for m in mice[1:]:
        common_idx = common_idx.intersection(per_mouse[m].index)

    aligned = {m: per_mouse[m].loc[common_idx] for m in mice if m in per_mouse}
    if not aligned:
        return pd.DataFrame(columns=["frame", "agent_id", "target_id"])

    if allowed_pairs is not None:
        pairs = [(a, t) for a, t in allowed_pairs if a in aligned and t in aligned]
        if not pairs:
            pairs = list(itertools.product(sorted(aligned), repeat=2))
    else:
        pairs = list(itertools.product(sorted(aligned), repeat=2))

    # Find a reference body part for relative features
    ref_parts = ["nose", "body_center", "neck", "tail_base", "ear_left", "ear_right"]

    rows = []
    for agent, target in pairs:
        A = aligned[agent]
        T = aligned[target]
        if A.empty or T.empty:
            continue

        feat = pd.concat([A, _make_target_prefixed(T)], axis=1)

        # Compute relative features from the first available reference part
        picked = None
        for p in ref_parts:
            ax, ay = f"x_{p}", f"y_{p}"
            if ax in A.columns and ay in A.columns and ax in T.columns and ay in T.columns:
                picked = p
                break

        if picked is not None:
            ax_v = A[f"x_{picked}"].to_numpy()
            ay_v = A[f"y_{picked}"].to_numpy()
            tx_v = T[f"x_{picked}"].to_numpy()
            ty_v = T[f"y_{picked}"].to_numpy()
            dx = tx_v - ax_v
            dy = ty_v - ay_v
            feat = feat.assign(
                rel_dx=dx, rel_dy=dy, rel_dist=np.sqrt(dx * dx + dy * dy)
            )
        else:
            feat = feat.assign(rel_dx=0.0, rel_dy=0.0, rel_dist=0.0)

        feat["agent_id"] = agent
        feat["target_id"] = target
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=["frame", "agent_id", "target_id"])

    out = pd.concat(rows, copy=False).reset_index(names="frame")
    id_cols = ["frame", "agent_id", "target_id"]
    feat_cols = [c for c in out.columns if c not in id_cols]
    return out[id_cols + feat_cols].astype(
        {c: np.float32 for c in feat_cols}, copy=False
    )


# ---------------------------------------------------------------------------
# 6. Robust scaling
# ---------------------------------------------------------------------------


def compute_scaler(
    df: pd.DataFrame, feat_cols: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute robust scaler parameters (median and IQR) from training data."""
    med = df[feat_cols].median().to_dict()
    q1 = df[feat_cols].quantile(0.25)
    q3 = df[feat_cols].quantile(0.75)
    iqr = (q3 - q1).replace(0, 1.0).to_dict()
    return {"med": med, "iqr": iqr, "clip_low": -5.0, "clip_high": 5.0}


def apply_scaler(
    df: pd.DataFrame, feat_cols: List[str], scaler: Dict
) -> pd.DataFrame:
    """Apply robust scaling: ``(x - median) / IQR``, clipped to [-5, 5]."""
    df = df.copy()
    med = pd.Series(scaler["med"], dtype=np.float32)
    iqr = pd.Series(scaler["iqr"], dtype=np.float32).replace(0, 1.0)
    low, high = float(scaler.get("clip_low", -5.0)), float(scaler.get("clip_high", 5.0))

    X = df[feat_cols].astype(np.float32)
    X = (X - med.reindex(feat_cols).fillna(0.0)) / iqr.reindex(feat_cols).fillna(1.0)
    X = X.clip(low, high).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df[feat_cols] = X.values
    return df


# ---------------------------------------------------------------------------
# 7. High-level: build features for one video
# ---------------------------------------------------------------------------


def build_video_features(
    lab_id: str,
    video_id: int,
    meta_row: pd.Series,
    tracking_dir: str,
) -> pd.DataFrame:
    """End-to-end feature pipeline for a single video.

    1. Load tracking parquet
    2. Drop unwanted body parts
    3. Pivot → per-mouse wide tables
    4. Normalize coordinates
    5. Fill missing + create masks
    6. Add velocities
    7. Build pairwise features

    Returns:
        DataFrame with columns ``[frame, agent_id, target_id, video_id, <features…>]``
    """
    path = os.path.join(tracking_dir, str(lab_id), f"{video_id}.parquet")
    track = load_tracking(path)
    track = track[~track["bodypart"].isin(DROP_BODY_PARTS)].copy()

    fps = float(meta_row.get("frames_per_second", 30.0))
    ppcm = meta_row.get("pix_per_cm_approx", np.nan)
    ppcm = float(ppcm) if pd.notna(ppcm) and float(ppcm) != 0 else np.nan
    w_pix = float(meta_row.get("video_width_pix", np.nan))
    h_pix = float(meta_row.get("video_height_pix", np.nan))

    # Fallback: estimate pix_per_cm from arena dimensions
    if not np.isfinite(ppcm):
        aw = meta_row.get("arena_width_cm", np.nan)
        ah = meta_row.get("arena_height_cm", np.nan)
        if (
            pd.notna(aw) and pd.notna(ah) and float(aw) > 0 and float(ah) > 0
            and np.isfinite(w_pix) and np.isfinite(h_pix)
        ):
            ppcm = ((w_pix / float(aw)) + (h_pix / float(ah))) / 2.0
    if not np.isfinite(ppcm) or ppcm == 0:
        ppcm = 1.0

    all_parts = sorted(track["bodypart"].unique().tolist())

    per_mouse = pivot_mouse_frame(track, all_parts)
    del track

    for m in list(per_mouse):
        pm = normalize_coords(per_mouse[m], ppcm, w_pix, h_pix)
        pm = fill_and_create_masks(pm)
        per_mouse[m] = add_velocities(pm, fps)

    feats = build_pairwise_features(per_mouse)
    del per_mouse

    feats["video_id"] = int(video_id)
    return feats
