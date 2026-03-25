from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    root_dir: Path = Path(__file__).resolve().parents[2]
    raw_data_path: Path = root_dir / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv.gz"
    processed_data_path: Path = root_dir / "data" / "processed" / "regional_demand.npz"
    output_dir: Path = root_dir / "outputs"

    horizon: int = 240
    intersections: int = 4
    history_steps: int = 12

    min_inflow: float = 2.0
    max_inflow: float = 18.0
    service_rate: float = 10.0
    transfer_ratio: float = 0.55

    train_episodes: int = 120
    val_episodes: int = 30
    test_episodes: int = 30

    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    d_model: int = 96
    nhead: int = 8
    num_layers: int = 3
    dropout: float = 0.1
