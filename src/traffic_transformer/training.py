from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from .config import ProjectConfig
from .model import TransformerPolicy


class RLSimulatorWrapper:
    def __init__(self, config: ProjectConfig, demand: np.ndarray):
        self.config = config
        self.demand = demand
        self.horizon = demand.shape[0]
        self.n = config.intersections
        self.service_rate = config.service_rate
        self.transfer_ratio = config.transfer_ratio

        self.t = 0
        self.queue = np.zeros((self.n, 2), dtype=np.float32)
        self.pending_inbound = np.zeros((self.n, 2), dtype=np.float32)
        self.history = deque(maxlen=config.history_steps)

    def _downstream(self, idx: int, direction: int) -> int:
        if direction == 0:
            mapping = {0: 2, 1: 3}
        else:
            mapping = {0: 1, 2: 3}
        return mapping.get(idx, -1)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.queue.fill(0.0)
        self.pending_inbound.fill(0.0)
        self.history.clear()
        for _ in range(self.config.history_steps):
            self.history.append(self.queue.copy())
        return np.stack(list(self.history), axis=0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        self.queue += self.demand[self.t] + self.pending_inbound
        self.pending_inbound.fill(0.0)

        departures = np.zeros((self.n, 2), dtype=np.float32)
        for i in range(self.n):
            a = action[i]
            green_dir = 1 if a >= 2 else 0
            duration_scale = 2.0 if a % 2 == 1 else 1.0
            departures[i, green_dir] = min(self.queue[i, green_dir], self.service_rate * duration_scale)

        self.queue -= departures

        # Apply transfer
        for i in range(self.n):
            for direction in (0, 1):
                moved = departures[i, direction] * self.transfer_ratio
                downstream = self._downstream(i, direction)
                if downstream != -1:
                    self.pending_inbound[downstream, direction] += moved

        self.t += 1
        done = self.t >= self.horizon
        
        # Dense Reward: Negative total vehicles waiting.
        # Encourages minimizing the overall queue length across the network.
        reward = -float(self.queue.sum()) 

        self.history.append(self.queue.copy())
        next_state = np.stack(list(self.history), axis=0)

        return next_state, reward, done


def train_transformer(
    config: ProjectConfig,
    train_episodes: np.ndarray,
    val_episodes: np.ndarray,
    device: str = "cpu",
) -> Path:
    
    print("\n[RL] Starting PPO Training Phase...")
    
    model = TransformerPolicy(
        intersections=config.intersections,
        feature_dim=2,
        history_steps=config.history_steps,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # Note: PPO typically uses a lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate) 
    
    model_path = config.output_dir / "transformer_policy.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    update_epochs = 4
    
    # Redefining epochs as PPO Iterations
    iterations = max(15, config.epochs) 
    episodes_per_iter = min(8, len(train_episodes))
    
    best_mean_reward = -float("inf")

    for it in range(1, iterations + 1):
        states, actions, log_probs, all_returns, all_advs = [], [], [], [], []
        
        model.eval()
        iter_rewards = []
        
        sampled_idxs = np.random.choice(len(train_episodes), size=episodes_per_iter, replace=False)
        
        for idx in sampled_idxs:
            env = RLSimulatorWrapper(config, train_episodes[idx])
            state = env.reset()
            
            ep_states, ep_actions, ep_logprobs, ep_raw_rewards, ep_values = [], [], [], [], []
            ep_total_reward = 0.0
            
            while True:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, val = model(state_t)
                    dist = Categorical(logits=logits.view(-1, 4))
                    action = dist.sample()
                    # sum over N nodes to get joint log prob
                    log_prob = dist.log_prob(action).sum() 
                    
                action_np = action.cpu().numpy()
                next_state, reward, done = env.step(action_np)
                
                ep_states.append(state)
                ep_actions.append(action_np)
                ep_logprobs.append(log_prob.item())
                
                # Scale reward down to keep Critic value targets bounded
                scaled_reward = reward / 100.0 
                ep_raw_rewards.append(scaled_reward)
                ep_values.append(val.item())
                
                ep_total_reward += reward
                state = next_state
                
                if done:
                    break
                    
            iter_rewards.append(ep_total_reward)
            
            # Generalized Advantage Estimation (GAE) Calculation
            returns = []
            advs = []
            gae = 0
            ep_values_full = ep_values + [0.0]  # Value for terminal state is 0
            
            for i in reversed(range(len(ep_raw_rewards))):
                delta = ep_raw_rewards[i] + gamma * ep_values_full[i+1] - ep_values_full[i]
                gae = delta + gamma * lam * gae
                advs.insert(0, gae)
                returns.insert(0, gae + ep_values_full[i])
                
            states.extend(ep_states)
            actions.extend(ep_actions)
            log_probs.extend(ep_logprobs)
            all_returns.extend(returns)
            all_advs.extend(advs)

        mean_iter_reward = float(np.mean(iter_rewards))
        
        # PPO Update phase
        model.train()
        
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        log_probs_t = torch.tensor(log_probs, dtype=torch.float32).to(device)
        returns_t = torch.tensor(all_returns, dtype=torch.float32).to(device)
        advs_t = torch.tensor(all_advs, dtype=torch.float32).to(device)
        
        # Advantage Normalization (crucial for PPO stability)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)
        
        dataset = TensorDataset(states_t, actions_t, log_probs_t, returns_t, advs_t)
        bs = min(config.batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        batches = 0
        
        for _ in range(update_epochs):
            for b_states, b_actions, b_log_probs, b_returns, b_advs in loader:
                logits, values = model(b_states)
                
                # Check categorical probabilities
                dist = Categorical(logits=logits.view(-1, 4))
                new_log_probs = dist.log_prob(b_actions.view(-1)).view(-1, config.intersections).sum(dim=-1) 
                
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - b_log_probs)
                
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * b_advs
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), b_returns)
                
                # Combined Loss (Actor + Critic - Entropy Bonus for exploration)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                batches += 1

        avg_actor_loss = total_actor_loss / max(1, batches)
        avg_critic_loss = total_critic_loss / max(1, batches)
        
        print(f"PPO Iter {it:02d}/{iterations} | Mean Rwd: {mean_iter_reward:7.1f} | Pi Loss: {avg_actor_loss:6.4f} | V Loss: {avg_critic_loss:6.4f}")
        
        if mean_iter_reward > best_mean_reward:
            best_mean_reward = mean_iter_reward
            torch.save(model.state_dict(), model_path)
            
    print(f"\n[RL] PPO Training Finished. Best Mean Reward: {best_mean_reward:.1f}")
    return model_path
