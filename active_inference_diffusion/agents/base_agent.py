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

from ..core.active_inference import ActiveInferenceCore
from ..encoders.visual_encoders import EncoderFactory, RandomShiftAugmentation
from ..utils.buffers import ReplayBuffer
from ..configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)


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
        
        # Create optimizers
        self._setup_optimizers()
        
        # Replay buffer
        self.replay_buffer = self._create_replay_buffer()
        
        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.exploration_noise = training_config.exploration_noise
        
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
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
            
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Process batch
        states = self._process_batch_observations(batch['observations'])
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = self._process_batch_observations(batch['next_observations'])
        dones = batch['dones'].to(self.device)
        
        metrics = {}
        
        # 1. Train dynamics model
        self.dynamics_optimizer.zero_grad()
        dynamics_metrics = self.active_inference.train_dynamics(
            states, actions, next_states, rewards
        )
        dynamics_loss = dynamics_metrics['total_loss']
        dynamics_loss_tensor = torch.tensor(dynamics_loss, requires_grad=True)
        dynamics_loss_tensor.backward()
        nn.utils.clip_grad_norm_(
            list(self.active_inference.dynamics_model.parameters()) +
            list(self.active_inference.reward_predictor.parameters()),
            self.config.gradient_clip
        )
        self.dynamics_optimizer.step()
        metrics.update(dynamics_metrics)
        
        # 2. Train value function
        self.value_optimizer.zero_grad()
        value_metrics = self.active_inference.update_value_function(
            states, rewards, next_states, dones
        )
        value_loss = torch.tensor(value_metrics['value_loss'], requires_grad=True)
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.active_inference.value_network.parameters(),
            self.config.gradient_clip
        )
        self.value_optimizer.step()
        metrics.update(value_metrics)
        
        # 3. Train score network (free energy)
        self.score_optimizer.zero_grad()
        free_energy, fe_info = self.active_inference.compute_free_energy_loss(
            states, states, actions  # Using states as observations
        )
        free_energy.backward()
        nn.utils.clip_grad_norm_(
            self.active_inference.score_network.parameters(),
            self.config.gradient_clip
        )
        self.score_optimizer.step()
        metrics['free_energy'] = free_energy.item()
        metrics['complexity'] = fe_info['complexity'].item()
        metrics['accuracy'] = fe_info['accuracy'].item()
        
        # 4. Train policy (minimize expected free energy)
        self.policy_optimizer.zero_grad()
        efe, efe_info = self.active_inference.compute_expected_free_energy(states)
        policy_loss = efe.mean()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(
            self.active_inference.policy_network.parameters(),
            self.config.gradient_clip
        )
        self.policy_optimizer.step()
        metrics['expected_free_energy'] = efe.mean().item()
        metrics['policy_loss'] = policy_loss.item()
        
        # Update precision
        self.active_inference.free_energy.update_precision(
            fe_info['complexity'],
            fe_info['accuracy']
        )
        metrics['precision'] = self.active_inference.free_energy.precision.item()
        
        return metrics
        
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

