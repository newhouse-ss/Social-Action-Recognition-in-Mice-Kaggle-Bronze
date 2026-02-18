"""Data loading and PyTorch Dataset for the CNN-Transformer pipeline.

Handles:
  - Loading train/test metadata CSVs
  - Parsing behavior-label triplets from metadata
  - Building frame-level label arrays from annotation parquets
  - PyTorch Datasets that yield sliding windows of features ± labels
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import ALL_ACTIONS, DROP_BODY_PARTS, TRAIN_LAB_ACTIONS


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_train_test(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv with basic preprocessing."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train["n_mice"] = 4 - train[
        ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]
    ].isna().sum(axis=1)

    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test["n_mice"] = 4 - test[
        ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]
    ].isna().sum(axis=1)

    return train, test


# ---------------------------------------------------------------------------
# Behavior label parsing
# ---------------------------------------------------------------------------


def parse_behaviors_labeled(raw: str) -> List[Tuple[str, str, str]]:
    """Parse the ``behaviors_labeled`` JSON string into (agent, target, action) triplets."""
    if not isinstance(raw, str) or pd.isna(raw):
        return []
    items = json.loads(raw)
    triplets = []
    for item in items:
        parts = item.replace("'", "").split(",")
        if len(parts) >= 3:
            triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return triplets


def build_active_map(meta_df: pd.DataFrame) -> Dict[int, Set[str]]:
    """Build ``{video_id: set of 'agent_id,target_id,action'}`` for probability masking."""
    amap: Dict[int, Set[str]] = {}
    for _, row in meta_df.iterrows():
        vid = int(row["video_id"])
        triplets = parse_behaviors_labeled(row.get("behaviors_labeled", ""))
        S: Set[str] = set()
        for agent_str, target_str, action in triplets:
            m_a = re.search(r"(\d+)", agent_str)
            if not m_a:
                continue
            ai = int(m_a.group(1))
            if target_str.lower() in ("self", "same"):
                ti = ai
            else:
                m_t = re.search(r"(\d+)", target_str)
                if not m_t:
                    continue
                ti = int(m_t.group(1))
            S.add(f"{ai},{ti},{action}")
        amap[vid] = S
    return amap


# ---------------------------------------------------------------------------
# Frame-level labels from annotation parquets
# ---------------------------------------------------------------------------


def load_annotations(
    annotation_dir: str, lab_id: str, video_id: int
) -> pd.DataFrame:
    """Load annotation parquet for one video."""
    path = os.path.join(annotation_dir, lab_id, f"{video_id}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=["agent_id", "target_id", "action", "start_frame", "stop_frame"]
        )
    return pd.read_parquet(path)


def build_frame_labels(
    n_frames: int,
    annotations: pd.DataFrame,
    agent_id: int,
    target_id: int,
    actions: List[str],
) -> np.ndarray:
    """Build ``[n_frames, n_actions]`` binary label array for one (agent, target) pair."""
    labels = np.zeros((n_frames, len(actions)), dtype=np.float32)
    action_to_idx = {a: i for i, a in enumerate(actions)}

    sub = annotations[
        (annotations["agent_id"] == agent_id)
        & (annotations["target_id"] == target_id)
    ]
    for _, row in sub.iterrows():
        action = str(row["action"]).strip().lower()
        idx = action_to_idx.get(action)
        if idx is None:
            continue
        start = int(row["start_frame"])
        stop = int(row["stop_frame"])
        labels[start:stop, idx] = 1.0

    return labels


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------


class BehaviorWindowDataset(Dataset):
    """Training dataset that yields fixed-length sliding windows.

    Each item is a tuple of:
        - ``features``: ``[window_size, n_features]`` float32 tensor
        - ``labels``:   ``[window_size, n_actions]``  float32 tensor
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = 128,
        window_step: int = 64,
    ):
        assert features.shape[0] == labels.shape[0]
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.window_size = window_size
        self.window_step = window_step

        n_frames = features.shape[0]
        self.starts = list(range(0, max(1, n_frames - window_size + 1), window_step))
        if self.starts and self.starts[-1] + window_size < n_frames:
            self.starts.append(n_frames - window_size)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        F = self.features.shape[0]

        feat = self.features[s : min(e, F)]
        lab = self.labels[s : min(e, F)]

        if feat.shape[0] < self.window_size:
            pad_len = self.window_size - feat.shape[0]
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode="edge")
            lab = np.pad(lab, ((0, pad_len), (0, 0)), mode="edge")

        return torch.from_numpy(feat), torch.from_numpy(lab)


class BehaviorInferenceDataset(Dataset):
    """Inference dataset — yields windows without labels."""

    def __init__(
        self,
        features: np.ndarray,
        window_size: int = 128,
        window_step: int = 64,
    ):
        self.features = features.astype(np.float32)
        self.window_size = window_size
        self.window_step = window_step
        n_frames = features.shape[0]
        self.starts = list(range(0, max(1, n_frames - window_size + 1), window_step))
        if self.starts and self.starts[-1] + window_size < n_frames:
            self.starts.append(n_frames - window_size)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        e = s + self.window_size
        F = self.features.shape[0]

        feat = self.features[s : min(e, F)]
        if feat.shape[0] < self.window_size:
            pad_len = self.window_size - feat.shape[0]
            feat = np.pad(feat, ((0, pad_len), (0, 0)), mode="edge")

        return torch.from_numpy(feat), s
