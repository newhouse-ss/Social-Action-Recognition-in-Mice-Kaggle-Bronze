"""CNN-Transformer model for frame-level mouse behavior classification.

Architecture overview:

    Input [B, T, F]
        │
    Input Projection  (Linear → GELU → Dropout)
        │
    Dilated CNN Backbone  (stack of 1D conv blocks, dilation = 1, 2, 4, 8, …)
        │
    Sinusoidal Positional Encoding
        │
    Transformer Encoder  (multi-head self-attention + feed-forward, pre-norm)
        │
    Classification Head  (LayerNorm → Linear)
        │
    Output [B, T, num_actions]   ← logits (apply sigmoid for probabilities)

The CNN captures local temporal patterns at multiple time scales through
dilated convolutions, while the Transformer models long-range dependencies
across the full window via self-attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (not learned).

    Injects frame-position information so the Transformer knows *where*
    in time each frame sits within the window.
    """

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CNNBlock(nn.Module):
    """One dilated 1-D convolution block with a residual connection.

    Processing path:  Conv1d → BatchNorm → GELU → Dropout
    Output:           residual + processed
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, dilation=dilation
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] → transpose for Conv1d → [B, D, T]
        residual = x
        h = x.transpose(1, 2)
        h = self.conv(h)
        h = self.norm(h)
        h = h.transpose(1, 2)  # back to [B, T, D]
        h = self.act(h)
        h = self.drop(h)
        return h + residual


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class CNNTransformer(nn.Module):
    """CNN-Transformer for frame-level multi-label behavior classification.

    Args:
        in_features:          Number of input features per frame.
        num_actions:          Number of behavior classes to predict.
        d_model:              Hidden dimension throughout the network.
        n_cnn_layers:         Number of stacked dilated CNN blocks.
        n_transformer_layers: Number of Transformer encoder layers.
        nhead:                Number of attention heads.
        dim_feedforward:      Feed-forward dimension inside each Transformer
                              layer (default: ``4 * d_model``).
        dropout:              Dropout probability.
        kernel_size:          Convolution kernel width.
    """

    def __init__(
        self,
        in_features: int,
        num_actions: int,
        d_model: int = 256,
        n_cnn_layers: int = 4,
        n_transformer_layers: int = 6,
        nhead: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.in_features = in_features
        self.num_actions = num_actions
        self.d_model = d_model

        # --- Input projection ---
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- Dilated CNN backbone ---
        self.cnn_blocks = nn.ModuleList(
            [
                CNNBlock(d_model, kernel_size, dilation=2**i, dropout=dropout)
                for i in range(n_cnn_layers)
            ]
        )

        # --- Positional encoding ---
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)

        # --- Transformer encoder (pre-norm for stable training) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape ``[batch_size, seq_len, in_features]``.

        Returns:
            Logits of shape ``[batch_size, seq_len, num_actions]``.
            Apply ``torch.sigmoid`` to obtain probabilities.
        """
        h = self.input_proj(x)

        for block in self.cnn_blocks:
            h = block(h)

        h = self.pos_enc(h)
        h = self.transformer(h)

        return self.head(h)
