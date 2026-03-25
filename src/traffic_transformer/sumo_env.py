from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import traci

from .config import ProjectConfig
from .simulator import SimulationResult


PolicyFn = Callable[[np.ndarray, int], np.ndarray]


class SumoTrafficSimulator:
    def __init__(self, config: ProjectConfig, demand: np.ndarray, sumocfg_path: str, gui: bool = False):
        self.config = config
        self.demand = demand  # Shape: (horizon, n_intersections, 2)
        self.horizon = demand.shape[0]
        self.n = config.intersections

        self.sumocfg_path = sumocfg_path
        self.gui = gui
        
        # We assume 4 intersections: "A0", "A1", "B0", "B1" based on netgenerate defaults
        self.tls_ids = ["A0", "A1", "B0", "B1"]
        
        # Action step length: How many SUMO seconds pass per one agent step
        self.step_length = 10
        self.yellow_duration = 3
        
        # Which phase ID corresponds to NS/EW green
        # This heavily depends on how the network is generated. 
        # Typically for basic crossings: Phase 0 is NS Green, Phase 2 is EW Green. (1 and 3 are yellows)
        self.ns_phase = 0
        self.ew_phase = 2

    def _get_queue_state(self) -> np.ndarray:
        queue = np.zeros((self.n, 2), dtype=np.float32)
        # Iterate over all lanes in the network
        lane_ids = traci.lane.getIDList()
        
        for lane in lane_ids:
            # For a 2x2 grid, we would identify if a lane goes NS or EW and approaches which junction
            # E.g., moving North (N) or South (S) is NS direction
            # This is a generic heuristic:
            node_id = traci.lane.getEdgeID(lane).split("_")[-1] # Usually ends with target node
            if node_id in self.tls_ids:
                idx = self.tls_ids.index(node_id)
                # Heuristic: simple check on compass direction of lane shape or ID
                # If Edge ID contains 'N' or 'S' -> NS direction, 'E' or 'W' -> EW
                edge_id = traci.lane.getEdgeID(lane).upper()
                direction = 0 if ('N' in edge_id or 'S' in edge_id) else 1
                
                queue[idx, direction] += traci.lane.getLastStepHaltingNumber(lane)

        return queue

    def _spawn_vehicles(self, t_step: int, current_sumo_time: float):
        """Spawn vehicles based on the demand array for the current step."""
        # limit demand range to avoid massive spam
        # demand shape is (horizon, nodes, 2)
        if t_step >= self.horizon:
            return
            
        current_demand = self.demand[t_step]
        for idx, tls in enumerate(self.tls_ids):
            ns_demand = int(current_demand[idx, 0])
            ew_demand = int(current_demand[idx, 1])
            
            # Spawn logic: we need predefined route IDs (e.g., 'route_ns_0', 'route_ew_0')
            # To be safely injected, we do a basic random distribution across the 10 seconds step
            # Here we just inject them using traci.vehicle.add if route exists
            try:
                for i in range(ns_demand):
                    veh_id = f"veh_{t_step}_{tls}_ns_{i}"
                    traci.vehicle.add(veh_id, f"route_ns_{idx}")
                for i in range(ew_demand):
                    veh_id = f"veh_{t_step}_{tls}_ew_{i}"
                    traci.vehicle.add(veh_id, f"route_ew_{idx}")
            except traci.exceptions.TraCIException:
                pass # Route not found, ignore in fallback mode


    def run(self, policy: PolicyFn, warmup: int = 5) -> SimulationResult:
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [binary, "-c", self.sumocfg_path, "--step-length", "1.0", "--no-warnings"]
        
        traci.start(cmd)
        
        cumulative_queue = 0.0
        throughput = 0.0
        
        try:
            for t in range(self.horizon):
                # Apply demand
                self._spawn_vehicles(t, traci.simulation.getTime())
                
                # Get Observation
                queue_state = self._get_queue_state()
                
                # Get Action from Policy
                action = policy(queue_state, t)
                
                # Handle Phase and Duration
                target_phases = []
                durations = []
                need_yellow = False
                for i, tls in enumerate(self.tls_ids):
                    a = action[i]
                    is_ew = a >= 2
                    is_long = a % 2 == 1
                    
                    target = self.ew_phase if is_ew else self.ns_phase
                    target_phases.append(target)
                    durations.append(20 if is_long else 10)
                    
                    current = traci.trafficlight.getPhase(tls)
                    if current != target and current not in [1, 3]:
                        traci.trafficlight.setPhase(tls, current + 1) # Set Yellow
                        need_yellow = True
                
                if need_yellow:
                    # Run 3 seconds of yellow
                    for _ in range(self.yellow_duration):
                        traci.simulationStep()
                
                # Set purely to target phases
                for i, tls in enumerate(self.tls_ids):
                    traci.trafficlight.setPhase(tls, target_phases[i])
                
                # Global step follows the maximum requested valid duration to keep history synchronous
                cycle_duration = max(durations)
                green_duration = cycle_duration - (self.yellow_duration if need_yellow else 0)
                if green_duration < 1:
                    green_duration = 1
                    
                for _ in range(green_duration):
                    traci.simulationStep()
                    
                # Collect Metrics
                if t >= warmup:
                    cumulative_queue += queue_state.sum()
                    throughput += traci.simulation.getArrivedNumber()

        finally:
            traci.close()

        effective_steps = max(1, self.horizon - warmup)
        avg_queue = cumulative_queue / effective_steps

        return SimulationResult(
            avg_queue=avg_queue,
            cumulative_queue=cumulative_queue,
            throughput=throughput,
        )
