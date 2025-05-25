"""
Environment wrappers
"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Optional, Tuple, Dict, Any


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [-1, 1]"""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Get bounds
        low = self.observation_space.low
        high = self.observation_space.high
        
        # Check if bounds are finite
        bounded = np.isfinite(low).all() and np.isfinite(high).all()
        
        if bounded:
            self.loc = (low + high) / 2.0
            self.scale = (high - low) / 2.0
        else:
            # Use running statistics
            self.loc = np.zeros(self.observation_space.shape)
            self.scale = np.ones(self.observation_space.shape)
            self.running_mean = np.zeros(self.observation_space.shape)
            self.running_var = np.ones(self.observation_space.shape)
            self.count = 1e-4
            
        # Update observation space
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation"""
        return (obs - self.loc) / (self.scale + 1e-8)


class ActionRepeat(gym.Wrapper):
    """Repeat actions for frame skipping"""
    
    def __init__(self, env: gym.Env, repeat: int = 1):
        super().__init__(env)
        self.repeat = repeat
        
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """Step environment with action repeat"""
        total_reward = 0.0
        
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
                
        return obs, total_reward, terminated, truncated, info


