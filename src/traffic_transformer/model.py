from __future__ import annotations

import torch
import torch.nn as nn


class TransformerPolicy(nn.Module):
    def __init__(
        self,
        intersections: int,
        feature_dim: int,
        history_steps: int,
        d_model: int = 96,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.intersections = intersections
        self.history_steps = history_steps

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.time_embed = nn.Embedding(history_steps, d_model)
        self.node_embed = nn.Embedding(intersections, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_model * 4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4), # 4 classes: [NS_10s, NS_20s, EW_10s, EW_20s]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        tokens = self.input_proj(x)

        t_ids = torch.arange(steps, device=x.device)
        n_ids = torch.arange(nodes, device=x.device)
        time_embed = self.time_embed(t_ids)[None, :, None, :]
        node_embed = self.node_embed(n_ids)[None, None, :, :]

        tokens = tokens + time_embed + node_embed
        tokens = tokens.reshape(batch, steps * nodes, -1)
        encoded = self.encoder(tokens)

        encoded = encoded.reshape(batch, steps, nodes, -1)
        last_step = encoded[:, -1]
        logits = self.head(last_step)
        return logits
