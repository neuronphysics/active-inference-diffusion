"""
Unified Active Inference + Diffusion Agent
Supports both state-based and pixel-based observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import gymnasium as gym

from ..core.active_inference import EMAModel
from ..encoder.visual_encoders import RandomShiftAugmentation
from ..encoder.state_encoders import StateEncoder, EncoderFactory
from ..utils.buffers import ReplayBuffer
from ..configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)

class RunningMeanStd:
    """Tracks running statistics for reward normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
    
class BaseActiveInferenceAgent(ABC):
    """Base class for Active Inference agents"""
    
    def __init__(
        self,
        env: gym.Env,
        config: ActiveInferenceConfig,
        training_config: TrainingConfig
    ):
        self.env = env
        self.config = config
        self.training_config = training_config
        self.device = torch.device(config.device)
        # Get environment dimensions
        self._setup_dimensions()
        
        # Create models
        self._build_models()
        # Add EMA for score network
        self.score_ema = EMAModel(
            self.active_inference.latent_score_network,
            decay=0.9999,
            device=self.device
        )       

        
        # Create optimizers
        self._setup_optimizers()
        
        # Replay buffer
        self.replay_buffer = self._create_replay_buffer()
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.exploration_noise = training_config.exploration_noise
        self.reward_normalizer = RunningMeanStd(shape=())
        
    @abstractmethod
    def _setup_dimensions(self):
        """Setup state/action dimensions"""
        pass
        
    @abstractmethod
    def _build_models(self):
        """Build agent models"""
        pass
        
    @abstractmethod
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer"""
        pass
        
    def _setup_optimizers(self):
        """Setup optimizers"""
        # Score network optimizer
        self.score_optimizer = torch.optim.Adam(
            self.active_inference.score_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.active_inference.policy_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Value optimizer
        self.value_optimizer = torch.optim.Adam(
            self.active_inference.value_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Dynamics optimizer
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.active_inference.dynamics_model.parameters()) +
            list(self.active_inference.reward_predictor.parameters()),
            lr=self.config.learning_rate
        )
        #Add epistemic optimizer
        self.epistemic_optimizer = torch.optim.Adam(
            self.active_inference.epistemic_estimator.parameters(),
            lr=self.config.learning_rate*0.1,
            weight_decay=1e-5
        )
        self.active_inference.epistemic_optimizer = self.epistemic_optimizer
        
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using active inference
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            info: Additional information
        """
        # Convert to tensor
        obs_tensor = self._process_observation(observation)
        obs_tensor = obs_tensor.to(self.device)
        
        # Get action from active inference
        with torch.no_grad():
            action_tensor, info = self.active_inference.act(
                obs_tensor,
                deterministic=deterministic
            )
            
        # Convert to numpy
        action = action_tensor.cpu().numpy().squeeze()
        
        # Add exploration noise if training
        if not deterministic and self.training and self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        return action, info
        
    @abstractmethod
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Process raw observation to tensor"""
        pass
        
    def train_step(self) -> Dict[str, float]:
        """Single training step"""
        pass

    @abstractmethod
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of observations"""
        pass
        
    def update_exploration(self):
        """Update exploration noise"""
        self.exploration_noise *= self.training_config.exploration_decay
        self.exploration_noise = max(
            self.exploration_noise,
            self.training_config.min_exploration
        )
        
    @property
    def training(self) -> bool:
        """Whether agent is in training mode"""
        return self.active_inference.training

