"""Training loop for the CNN-Transformer model.

Provides:
  - ``train_one_epoch``  – single-epoch training with mixed-precision
  - ``validate``         – evaluation pass returning average loss
  - ``train_model``      – full training driver with learning-rate schedule,
                           early stopping, and checkpoint saving
"""

from __future__ import annotations

import gc
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .config import TrainConfig
from .model import CNNTransformer


# ---------------------------------------------------------------------------
# Single epoch helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Run one training epoch. Returns average BCE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Run validation pass. Returns average BCE loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and torch.cuda.is_available():
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
        else:
            logits = model(features)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------


def train_model(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    in_features: int,
    num_actions: int,
    cfg: TrainConfig,
    save_dir: str = "checkpoints",
    verbose: bool = True,
) -> CNNTransformer:
    """Train a CNN-Transformer model from scratch.

    Args:
        train_loader: Training DataLoader yielding (features, labels) windows.
        val_loader:   Optional validation DataLoader.
        in_features:  Number of input feature columns.
        num_actions:  Number of behavior classes.
        cfg:          Training configuration dataclass.
        save_dir:     Directory to save model checkpoints.
        verbose:      Print progress to stdout.

    Returns:
        The trained ``CNNTransformer`` model (on CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train] device={device}, epochs={cfg.epochs}, lr={cfg.lr}")

    mcfg = cfg.model
    model = CNNTransformer(
        in_features=in_features,
        num_actions=num_actions,
        d_model=mcfg.d_model,
        n_cnn_layers=mcfg.n_cnn_layers,
        n_transformer_layers=mcfg.n_transformer_layers,
        nhead=mcfg.nhead,
        dim_feedforward=mcfg.dim_feedforward,
        dropout=mcfg.dropout,
        kernel_size=mcfg.kernel_size,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Learning rate schedule: linear warmup → cosine annealing
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs - cfg.warmup_epochs))
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
    )

    use_amp = cfg.mixed_precision and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        scheduler.step()

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, use_amp)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        if verbose:
            print(
                f"  epoch {epoch+1:3d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr_now:.2e} | {dt:.1f}s"
            )

        # Checkpoint best model
        if val_loader is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pt"))

    # Load best checkpoint if validation was used
    if val_loader is not None and best_epoch > 0:
        best_path = os.path.join(save_dir, "best_model.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            if verbose:
                print(f"[train] loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    model.cpu()
    return model
