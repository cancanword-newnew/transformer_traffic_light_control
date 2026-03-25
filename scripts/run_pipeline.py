from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from traffic_transformer.config import ProjectConfig
from traffic_transformer.dataset import download_dataset, load_split, preprocess_dataset
from traffic_transformer.evaluate import evaluate_model
from traffic_transformer.training import train_transformer


def main() -> None:
    config = ProjectConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Downloading dataset...")
    raw_path = download_dataset(config)
    print(f"Saved raw data: {raw_path}")

    print("[2/4] Preprocessing to regional demand episodes...")
    processed_path = preprocess_dataset(config)
    print(f"Saved processed data: {processed_path}")

    train_episodes, val_episodes, test_episodes = load_split(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/4] Training Transformer policy on {device}...")
    model_path = train_transformer(config, train_episodes, val_episodes, device=device)
    print(f"Saved model: {model_path}")

    print("[4/4] Evaluating policies...")
    aggregate = evaluate_model(config, model_path, test_episodes, device=device)

    print("=== Evaluation Summary ===")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
