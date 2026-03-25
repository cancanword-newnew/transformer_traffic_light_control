from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from .config import ProjectConfig

PolicyFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class SimulationResult:
    avg_queue: float
    cumulative_queue: float
    throughput: float


class RegionTrafficSimulator:
    def __init__(self, config: ProjectConfig, demand: np.ndarray):
        self.config = config
        self.demand = demand
        self.horizon = demand.shape[0]
        self.n = config.intersections
        self.service_rate = config.service_rate
        self.transfer_ratio = config.transfer_ratio

    def _downstream(self, idx: int, direction: int) -> int:
        if direction == 0:
            mapping = {0: 2, 1: 3}
        else:
            mapping = {0: 1, 2: 3}
        return mapping.get(idx, -1)

    def run(self, policy: PolicyFn, warmup: int = 0) -> SimulationResult:
        queue = np.zeros((self.n, 2), dtype=np.float32)
        pending_inbound = np.zeros((self.n, 2), dtype=np.float32)

        cumulative_queue = 0.0
        throughput = 0.0

        for t in range(self.horizon):
            queue += self.demand[t] + pending_inbound
            pending_inbound.fill(0.0)

            action = policy(queue.copy(), t)
            departures = np.zeros((self.n, 2), dtype=np.float32)

            for i in range(self.n):
                green_dir = int(action[i])
                departures[i, green_dir] = min(queue[i, green_dir], self.service_rate)

            queue -= departures

            for i in range(self.n):
                for direction in (0, 1):
                    moved = departures[i, direction] * self.transfer_ratio
                    downstream = self._downstream(i, direction)
                    if downstream == -1:
                        throughput += departures[i, direction]
                    else:
                        pending_inbound[downstream, direction] += moved
                        throughput += departures[i, direction] - moved

            if t >= warmup:
                cumulative_queue += float(queue.sum())

        effective_steps = max(1, self.horizon - warmup)
        avg_queue = cumulative_queue / effective_steps

        return SimulationResult(
            avg_queue=avg_queue,
            cumulative_queue=cumulative_queue,
            throughput=throughput,
        )


def fixed_time_policy(queue: np.ndarray, t: int, cycle: int = 6) -> np.ndarray:
    phase = (t // cycle) % 2
    return np.full((queue.shape[0],), phase, dtype=np.int64)


def max_pressure_policy(queue: np.ndarray, _t: int) -> np.ndarray:
    ns_pressure = queue[:, 0]
    ew_pressure = queue[:, 1]
    return (ew_pressure > ns_pressure).astype(np.int64)


def evaluate_policies(config: ProjectConfig, demand_episode: np.ndarray, model_policy: PolicyFn) -> Dict[str, SimulationResult]:
    sim = RegionTrafficSimulator(config, demand_episode)

    return {
        "fixed_time": sim.run(lambda q, t: fixed_time_policy(q, t)),
        "max_pressure": sim.run(max_pressure_policy),
        "transformer": sim.run(model_policy),
    }
