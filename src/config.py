from __future__ import annotations

import os
import random
import shutil

import numpy as np

SEED = 1234
VERBOSE = True
DEFAULT_DATA_DIR = "/kaggle/input/MABe-mouse-behavior-detection"
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


def detect_gpu() -> bool:
    return ("KAGGLE_KERNEL_RUN_TYPE" in os.environ) and (shutil.which("nvidia-smi") is not None)


USE_GPU = detect_gpu()


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_dir(data_dir: str | None) -> str:
    if data_dir:
        return data_dir
    return os.environ.get("MABE_DATA_DIR", DEFAULT_DATA_DIR)


__all__ = ["DEFAULT_DATA_DIR", "DROP_BODY_PARTS", "SEED", "USE_GPU", "resolve_data_dir", "set_seed"]
