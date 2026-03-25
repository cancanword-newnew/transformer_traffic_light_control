import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from traffic_transformer.config import ProjectConfig
from traffic_transformer.dataset import load_split
from traffic_transformer.sumo_env import SumoTrafficSimulator
from traffic_transformer.simulator import fixed_time_policy, max_pressure_policy
from traffic_transformer.evaluate import TransformerPolicyRunner
from traffic_transformer.model import TransformerPolicy


def evaluate_sumo():
    config = ProjectConfig()
    _, _, test_episodes = load_split(config)
    
    # Path to where SUMO generated the config
    sumocfg_path = str(ROOT / "data" / "sumo" / "sim.sumocfg")
    
    if not Path(sumocfg_path).exists():
        print("SUMO Configuration not found! Please run 'python scripts/generate_sumo_network.py' first.")
        return

    # Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = config.output_dir / "transformer_policy.pt"
    if not model_path.exists():
        print(f"Model not found at {model_path}! Please run 'python scripts/run_pipeline.py' first.")
        return
        
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

    print("\n--- Evaluating Fixed Time on SUMO ---")
    sim_ft = SumoTrafficSimulator(config, test_episodes[0], sumocfg_path, gui=False)
    res_ft = sim_ft.run(lambda q, t: fixed_time_policy(q, t))
    print(f"Fixed Time - Avg Queue: {res_ft.avg_queue:.2f}, Throughput: {res_ft.throughput}")

    print("\n--- Evaluating Max Pressure on SUMO ---")
    sim_mp = SumoTrafficSimulator(config, test_episodes[0], sumocfg_path, gui=False)
    res_mp = sim_mp.run(max_pressure_policy)
    print(f"Max Pressure - Avg Queue: {res_mp.avg_queue:.2f}, Throughput: {res_mp.throughput}")

    print("\n--- Evaluating Transformer on SUMO ---")
    runner = TransformerPolicyRunner(config, model, device)
    sim_tr = SumoTrafficSimulator(config, test_episodes[0], sumocfg_path, gui=False) # GUI can be set to True here!
    res_tr = sim_tr.run(runner)
    print(f"Transformer - Avg Queue: {res_tr.avg_queue:.2f}, Throughput: {res_tr.throughput}")


if __name__ == "__main__":
    evaluate_sumo()
