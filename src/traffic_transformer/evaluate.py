from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import ProjectConfig
from .model import TransformerPolicy
from .simulator import SimulationResult, evaluate_policies


class TransformerPolicyRunner:
    def __init__(self, config: ProjectConfig, model: TransformerPolicy, device: str = "cpu"):
        self.config = config
        self.model = model
        self.device = device
        self.history: deque[np.ndarray] = deque(maxlen=config.history_steps)

    def reset(self) -> None:
        self.history.clear()

    def __call__(self, queue: np.ndarray, _t: int) -> np.ndarray:
        self.history.append(queue.copy())
        if len(self.history) < self.config.history_steps:
            while len(self.history) < self.config.history_steps:
                self.history.appendleft(self.history[0].copy())

        window = np.stack(list(self.history), axis=0)
        x = torch.from_numpy(window[None].astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            action = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        return action.astype(np.int64)


def _aggregate(results: List[Dict[str, SimulationResult]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = results[0].keys()

    for key in keys:
        avg_queue = float(np.mean([r[key].avg_queue for r in results]))
        cumulative_queue = float(np.mean([r[key].cumulative_queue for r in results]))
        throughput = float(np.mean([r[key].throughput for r in results]))
        out[key] = {
            "avg_queue": avg_queue,
            "cumulative_queue": cumulative_queue,
            "throughput": throughput,
        }
    return out


def evaluate_model(config: ProjectConfig, model_path: Path, test_episodes: np.ndarray, device: str = "cpu") -> Dict[str, Dict[str, float]]:
    model = TransformerPolicy(
        intersections=config.intersections,
        feature_dim=2,
        history_steps=config.history_steps,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_results = []
    for episode in test_episodes:
        runner = TransformerPolicyRunner(config, model, device)
        runner.reset()
        policy_results = evaluate_policies(config, episode, model_policy=runner)
        all_results.append(policy_results)

    aggregate = _aggregate(all_results)

    output_json = config.output_dir / "evaluation_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    _plot_results(config, aggregate)
    return aggregate


def _plot_results(config: ProjectConfig, aggregate: Dict[str, Dict[str, float]]) -> None:
    labels = list(aggregate.keys())
    avg_queue = [aggregate[k]["avg_queue"] for k in labels]
    throughput = [aggregate[k]["throughput"] for k in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, avg_queue, color=["#999999", "#4C78A8", "#F58518"])
    axes[0].set_title("Average Queue (lower better)")
    axes[0].set_ylabel("vehicles")

    axes[1].bar(labels, throughput, color=["#999999", "#4C78A8", "#F58518"])
    axes[1].set_title("Throughput (higher better)")
    axes[1].set_ylabel("vehicles")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(config.output_dir / "evaluation_plot.png", dpi=150)
    plt.close(fig)
