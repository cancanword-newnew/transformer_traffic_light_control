from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

from .config import ProjectConfig


DEFAULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"


def download_dataset(config: ProjectConfig, url: str = DEFAULT_DATA_URL) -> Path:
    config.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    if config.raw_data_path.exists():
        return config.raw_data_path

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    config.raw_data_path.write_bytes(response.content)
    return config.raw_data_path


def _make_episode_demand(volume: np.ndarray, config: ProjectConfig, rng: np.random.Generator) -> np.ndarray:
    base = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    inflow = config.min_inflow + base * (config.max_inflow - config.min_inflow)

    pattern = np.array(
        [
            [1.00, 0.80],
            [0.85, 1.05],
            [1.10, 0.90],
            [0.90, 1.00],
        ],
        dtype=np.float32,
    )

    noise = rng.normal(loc=1.0, scale=0.08, size=(config.horizon, config.intersections, 2)).astype(np.float32)
    demand = inflow[:, None, None].astype(np.float32) * pattern[None, :, :] * noise
    demand = np.clip(demand, a_min=0.0, a_max=None)
    return demand


def preprocess_dataset(config: ProjectConfig, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)

    dataframe = pd.read_csv(config.raw_data_path, compression="infer")
    dataframe["date_time"] = pd.to_datetime(dataframe["date_time"])
    dataframe = dataframe.sort_values("date_time")

    hourly = dataframe.set_index("date_time")["traffic_volume"].astype(float).resample("h").mean().interpolate()
    values = hourly.values

    total_episodes = config.train_episodes + config.val_episodes + config.test_episodes
    required = total_episodes * config.horizon
    if len(values) < required:
        repeats = int(np.ceil(required / len(values)))
        values = np.tile(values, repeats)

    values = values[:required]
    values = values.reshape(total_episodes, config.horizon)

    all_demand = np.stack([_make_episode_demand(episode, config, rng) for episode in values], axis=0)

    train_end = config.train_episodes
    val_end = train_end + config.val_episodes

    split: Dict[str, np.ndarray] = {
        "train": all_demand[:train_end],
        "val": all_demand[train_end:val_end],
        "test": all_demand[val_end:],
    }

    config.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.processed_data_path, **split)
    return config.processed_data_path


def load_split(config: ProjectConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(config.processed_data_path)
    return data["train"], data["val"], data["test"]
