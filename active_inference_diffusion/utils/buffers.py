# ===== active_inference_diffusion/utils/buffers.py =====
"""
Replay buffer implementations
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import lz4.frame


class ReplayBuffer:
    """
    Efficient replay buffer with optional compression
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        optimize_memory: bool = False
    ):
        self.capacity = capacity
        self.device = device
        self.optimize_memory = optimize_memory
        self.pos = 0
        self.size = 0
        
        # Allocate memory
        if optimize_memory and len(obs_shape) == 3:  # Pixel observations
            self.observations = [None] * capacity
            self.next_observations = [None] * capacity
            self.compress = True
        else:
            self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
            self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
            self.compress = False
            
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition to buffer"""
        if self.compress:
            # Compress pixel observations
            self.observations[self.pos] = self._compress(obs)
            self.next_observations[self.pos] = self._compress(next_obs)
        else:
            # Store directly
            self.observations[self.pos] = torch.from_numpy(obs)
            self.next_observations[self.pos] = torch.from_numpy(next_obs)
            
        self.actions[self.pos] = torch.from_numpy(action)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions"""
        indices = np.random.randint(0, self.size, batch_size)
        
        if self.compress:
            # Decompress observations
            obs = torch.stack([
                torch.from_numpy(self._decompress(self.observations[i]))
                for i in indices
            ])
            next_obs = torch.stack([
                torch.from_numpy(self._decompress(self.next_observations[i]))
                for i in indices
            ])
        else:
            obs = self.observations[indices]
            next_obs = self.next_observations[indices]
            
        return {
            'observations': obs,
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': next_obs,
            'dones': self.dones[indices]
        }
        
    def _compress(self, data: np.ndarray) -> bytes:
        """Compress numpy array"""
        return lz4.frame.compress(data.tobytes())
        
    def _decompress(self, data: bytes) -> np.ndarray:
        """Decompress to numpy array"""
        decompressed = lz4.frame.decompress(data)
        # Infer shape from data size
        return np.frombuffer(decompressed, dtype=np.uint8).reshape(-1)
        
    def __len__(self):
        return self.size

