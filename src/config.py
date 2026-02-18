"""Central configuration for the CNN-Transformer mouse behavior pipeline.

Contains hyperparameters, constants, and per-lab action definitions used
throughout the project.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 1234


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "/kaggle/input/MABe-mouse-behavior-detection"


def resolve_data_dir(data_dir: str | None) -> str:
    if data_dir:
        return data_dir
    return os.environ.get("MABE_DATA_DIR", DEFAULT_DATA_DIR)


# ---------------------------------------------------------------------------
# Body part filtering
# ---------------------------------------------------------------------------

DROP_BODY_PARTS = [
    "headpiece_bottombackleft",
    "headpiece_bottombackright",
    "headpiece_bottomfrontleft",
    "headpiece_bottomfrontright",
    "headpiece_topbackleft",
    "headpiece_topbackright",
    "headpiece_topfrontleft",
    "headpiece_topfrontright",
    "spine_1",
    "spine_2",
    "tail_middle_1",
    "tail_middle_2",
    "tail_midpoint",
]

# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------

SELF_ACTIONS = [
    "rear", "selfgroom", "genitalgroom", "rest", "climb", "dig",
    "run", "freeze", "biteobject", "exploreobject", "huddle",
]

PAIR_ACTIONS = [
    "approach", "attack", "avoid", "chase", "chaseattack", "submit",
    "shepherd", "disengage", "mount", "sniff", "sniffgenital",
    "dominancemount", "sniffbody", "sniffface", "attemptmount",
    "intromit", "escape", "reciprocalsniff", "dominance", "allogroom",
    "ejaculate", "defend", "dominancegroom", "flinch", "follow", "tussle",
]

ALL_ACTIONS: List[str] = sorted(set(SELF_ACTIONS + PAIR_ACTIONS))

TRAIN_LAB_ACTIONS: Dict[str, List[str]] = {
    "AdaptableSnail": [
        "approach", "attack", "avoid", "chase", "chaseattack", "rear", "submit",
    ],
    "BoisterousParrot": ["shepherd"],
    "CRIM13": [
        "approach", "attack", "disengage", "mount", "rear", "selfgroom", "sniff",
    ],
    "CalMS21_supplemental": [
        "approach", "attemptmount", "attack", "dominancemount", "intromit",
        "mount", "sniff", "sniffbody", "sniffgenital", "sniffface",
    ],
    "CalMS21_task1": [
        "approach", "attack", "genitalgroom", "intromit", "mount",
        "sniff", "sniffbody", "sniffgenital", "sniffface",
    ],
    "CalMS21_task2": ["attack", "mount", "sniff"],
    "CautiousGiraffe": [
        "chase", "escape", "reciprocalsniff", "sniff", "sniffbody", "sniffgenital",
    ],
    "DeliriousFly": ["attack", "dominance", "sniff"],
    "ElegantMink": [
        "allogroom", "attack", "attemptmount", "ejaculate",
        "intromit", "mount", "sniff",
    ],
    "GroovyShrew": [
        "attemptmount", "climb", "defend", "dig", "escape", "rear",
        "rest", "run", "selfgroom", "sniff", "sniffgenital", "approach",
    ],
    "InvincibleJellyfish": [
        "allogroom", "attack", "dig", "dominancegroom", "escape",
        "selfgroom", "sniff", "sniffgenital",
    ],
    "JovialSwallow": ["attack", "chase", "sniff"],
    "LyricalHare": [
        "approach", "attack", "defend", "escape", "freeze", "rear", "sniff",
    ],
    "NiftyGoldfinch": [
        "approach", "attack", "biteobject", "chase", "climb", "defend",
        "dig", "escape", "exploreobject", "flinch", "follow", "rear",
        "run", "selfgroom", "sniff", "sniffgenital", "sniffface", "tussle",
    ],
    "PleasantMeerkat": ["attack", "chase", "escape", "follow"],
    "ReflectiveManatee": ["attack", "sniff"],
    "SparklingTapir": ["attack", "defend", "escape", "mount"],
    "TranquilPanther": [
        "intromit", "mount", "rear", "selfgroom", "sniff", "sniffgenital",
    ],
    "UppityFerret": ["huddle", "reciprocalsniff", "sniffgenital"],
}

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """CNN-Transformer architecture hyperparameters."""
    d_model: int = 256
    n_cnn_layers: int = 4
    n_transformer_layers: int = 6
    nhead: int = 8
    dim_feedforward: int | None = None  # defaults to 4 * d_model
    dropout: float = 0.1
    kernel_size: int = 5


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training loop settings."""
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    window_size: int = 128
    window_step: int = 64
    seed: int = SEED
    mixed_precision: bool = True
    warmup_epochs: int = 3
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------


@dataclass
class InferConfig:
    """Inference and post-processing settings."""
    window_size: int = 128
    window_step: int = 64
    batch_windows: int = 2048
    smooth_win: int = 5
    min_event_len: int = 3
    max_gap: int = 2
    default_threshold: float = 0.5
