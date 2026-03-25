from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import ProjectConfig
from .model import TransformerPolicy
from .simulator import max_pressure_policy


def _rollout_expert(config: ProjectConfig, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = config.intersections
    horizon = demand.shape[0]
    queue = np.zeros((n, 2), dtype=np.float32)
    pending = np.zeros((n, 2), dtype=np.float32)

    states = []
    labels = []

    for t in range(horizon):
        queue += demand[t] + pending
        pending.fill(0.0)

        action = max_pressure_policy(queue, t)
        states.append(queue.copy())
        labels.append(action.copy())

        departures = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            a = action[i]
            green_dir = 1 if a >= 2 else 0
            duration_scale = 2.0 if a % 2 == 1 else 1.0
            departures[i, green_dir] = min(queue[i, green_dir], config.service_rate * duration_scale)

        queue -= departures

        for i in range(n):
            moved_ns = departures[i, 0] * config.transfer_ratio
            moved_ew = departures[i, 1] * config.transfer_ratio

            if i == 0:
                pending[2, 0] += moved_ns
                pending[1, 1] += moved_ew
            elif i == 1:
                pending[3, 0] += moved_ns
            elif i == 2:
                pending[3, 1] += moved_ew

    return np.stack(states), np.stack(labels)


def _build_window_dataset(config: ProjectConfig, episodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    windows = []
    targets = []

    for episode in episodes:
        states, labels = _rollout_expert(config, episode)
        history = deque(maxlen=config.history_steps)

        for t in range(states.shape[0]):
            history.append(states[t])
            if len(history) < config.history_steps:
                continue

            window = np.stack(list(history), axis=0)
            windows.append(window)
            targets.append(labels[t])

    x = np.stack(windows).astype(np.float32)
    y = np.stack(targets).astype(np.int64)
    return x, y


def train_transformer(
    config: ProjectConfig,
    train_episodes: np.ndarray,
    val_episodes: np.ndarray,
    device: str = "cpu",
) -> Path:
    train_x, train_y = _build_window_dataset(config, train_episodes)
    val_x, val_y = _build_window_dataset(config, val_episodes)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = TransformerPolicy(
        intersections=config.intersections,
        feature_dim=2,
        history_steps=config.history_steps,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    model_path = config.output_dir / "transformer_policy.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits.reshape(-1, 4), batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss = F.cross_entropy(logits.reshape(-1, 4), batch_y.reshape(-1))
                val_loss += float(loss.item())

                pred = logits.argmax(dim=-1)
                val_correct += int((pred == batch_y).sum().item())
                val_total += int(batch_y.numel())

        val_loss /= max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

    return model_path
