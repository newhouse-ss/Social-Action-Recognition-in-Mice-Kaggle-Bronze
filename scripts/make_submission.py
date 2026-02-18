"""Generate a submission CSV using trained CNN-Transformer model(s).

Usage::

    python scripts/make_submission.py \\
        --data-dir data/ \\
        --checkpoint checkpoints/best_model.pt \\
        --output submission.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch

from src.config import (
    ALL_ACTIONS,
    InferConfig,
    ModelConfig,
    SEED,
    TRAIN_LAB_ACTIONS,
    resolve_data_dir,
    set_seed,
)
from src.data import build_active_map, load_train_test
from src.features import apply_scaler, build_video_features
from src.inference import ensemble_predict, mask_probs, smooth_probs
from src.model import CNNTransformer
from src.postprocess import format_submission, probs_to_intervals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CNN-Transformer inference and create submission.csv.")
    p.add_argument("--data-dir", default=None, help="Root data directory.")
    p.add_argument("--checkpoint", nargs="+", required=True, help="One or more model checkpoint paths (.pt).")
    p.add_argument("--scaler", default=None, help="Path to scaler .npz (optional).")
    p.add_argument("--output", default="submission.csv", help="Output CSV path.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    verbose = not args.quiet
    cfg = InferConfig()

    if verbose:
        print(f"[submit] device={device}, data_dir={data_dir}")
        print(f"[submit] checkpoints: {args.checkpoint}")

    # --- Load metadata ---
    train_meta, test_meta = load_train_test(data_dir)
    active_map = build_active_map(test_meta)
    test_list = (
        test_meta.drop_duplicates(subset=["lab_id", "video_id"], keep="first")
        [["lab_id", "video_id"]]
        .values.tolist()
    )

    # --- Build features for all test videos ---
    tracking_dir = os.path.join(data_dir, "test_tracking")
    all_feats = []
    for lab_id, video_id in test_list:
        meta_row = test_meta[test_meta["video_id"] == video_id].iloc[0]
        try:
            feats = build_video_features(lab_id, int(video_id), meta_row, tracking_dir)
            all_feats.append(feats)
            if verbose:
                print(f"  built feats: lab={lab_id}, vid={video_id}, rows={len(feats)}")
        except Exception as e:
            if verbose:
                print(f"  [skip] lab={lab_id}, vid={video_id}: {e}")
    if not all_feats:
        print("[submit] ERROR: no features built")
        return

    feats_df = pd.concat(all_feats, ignore_index=True)
    del all_feats
    gc.collect()

    id_cols = ["video_id", "frame", "agent_id", "target_id"]
    feat_cols = [c for c in feats_df.columns if c not in id_cols]

    # --- Apply scaler if provided ---
    if args.scaler and os.path.exists(args.scaler):
        scaler_data = dict(np.load(args.scaler, allow_pickle=True))
        scaler = {k: v.item() if hasattr(v, "item") else v for k, v in scaler_data.items()}
        feats_df = apply_scaler(feats_df, feat_cols, scaler)

    # --- Load models ---
    in_features = len(feat_cols)
    mcfg = ModelConfig()
    models = []
    for ckpt_path in args.checkpoint:
        model = CNNTransformer(
            in_features=in_features,
            num_actions=len(ALL_ACTIONS),
            d_model=mcfg.d_model,
            n_cnn_layers=mcfg.n_cnn_layers,
            n_transformer_layers=mcfg.n_transformer_layers,
            nhead=mcfg.nhead,
            dim_feedforward=mcfg.dim_feedforward,
            dropout=0.0,
            kernel_size=mcfg.kernel_size,
        )
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        if verbose:
            print(f"  loaded checkpoint: {ckpt_path}")

    # --- Ensemble inference ---
    t0 = time.time()
    probs = ensemble_predict(models, ALL_ACTIONS, feats_df, feat_cols, cfg, device=device)
    if verbose:
        print(f"[submit] inference done in {time.time()-t0:.1f}s, probs rows={len(probs)}")

    # --- Post-process ---
    probs = smooth_probs(probs, ALL_ACTIONS, win=cfg.smooth_win)
    probs = mask_probs(probs, ALL_ACTIONS, active_map)
    intervals = probs_to_intervals(
        probs, ALL_ACTIONS,
        min_len=cfg.min_event_len, max_gap=cfg.max_gap,
        default_threshold=cfg.default_threshold,
    )
    submission = format_submission(intervals)
    submission.to_csv(args.output, index=False)

    if verbose:
        print(f"[submit] wrote {args.output} with {len(submission)} rows")


if __name__ == "__main__":
    main()
