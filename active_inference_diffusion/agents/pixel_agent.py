"""
Pixel-based Active Inference agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional

from .base_agent import BaseActiveInferenceAgent
from ..core.active_inference import ActiveInferenceCore
from ..encoder.visual_encoders import RandomShiftAugmentation
from ..encoder.state_encoders import EncoderFactory
from ..utils.buffers import ReplayBuffer
from ..configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)
class PixelBasedAgent(BaseActiveInferenceAgent):
    """Agent for pixel-based observations"""
    
    def __init__(
        self,
        env: gym.Env,
        config: ActiveInferenceConfig,
        training_config: TrainingConfig,
        pixel_config: PixelObservationConfig
    ):
        self.pixel_config = pixel_config
        super().__init__(env, config, training_config)
        
        # Augmentation
        self.augmentation = RandomShiftAugmentation(
            pad=pixel_config.random_shift_pad
        )
        
    def _setup_dimensions(self):
        """Setup dimensions for pixel observations"""
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Get dimensions
        if isinstance(self.obs_space, gym.spaces.Box):
            # Handle frame stacking
            obs_shape = self.obs_space.shape
            if len(obs_shape) == 3:  # (C, H, W)
                self.obs_shape = obs_shape
            else:
                raise ValueError(f"Unexpected observation shape: {obs_shape}")
        else:
            raise ValueError(f"Unsupported observation space: {type(self.obs_space)}")
            
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
            
        # Update config
        self.config.action_dim = self.action_dim
        
    def _build_models(self):
        """Build models for pixel observations"""
        # Visual encoder
        self.encoder = EncoderFactory.create_encoder(
            encoder_type=self.pixel_config.encoder_type,
            obs_shape=self.pixel_config.image_shape,
            feature_dim=self.pixel_config.encoder_feature_dim,
            frame_stack=self.pixel_config.frame_stack
        )
        
        # Active inference core
        self.active_inference = PixelActiveInference(
            state_dim=self.pixel_config.encoder_feature_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            config=self.config,
            encoder=self.encoder,
            augmentation=self.augmentation if self.pixel_config.augmentation else None
        )
        
        # Move to device
        self.active_inference = self.active_inference.to(self.device)
        
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer for pixels"""
        return ReplayBuffer(
            capacity=self.training_config.buffer_size,
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            device=self.device,
            optimize_memory=True  # Use compression for pixels
        )
        
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert pixel observation to tensor"""
        # Ensure channel-first format
        if observation.shape[-1] == 3 and len(observation.shape) == 3:
            observation = np.transpose(observation, (2, 0, 1))
            
        return torch.FloatTensor(observation).unsqueeze(0)
        
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of pixel observations"""
        # Normalize to [0, 1] if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
            
        return observations.to(self.device)
        
    def train_step(self) -> Dict[str, float]:
        """Extended training step with contrastive learning"""
        metrics = super().train_step()
        
        if len(self.replay_buffer) < self.config.batch_size:
            return metrics
            
        # Additional pixel-specific training
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        obs = self._process_batch_observations(batch['observations'])
        next_obs = self._process_batch_observations(batch['next_observations'])
        actions = batch['actions'].to(self.device)
        
        # Contrastive loss for representation learning
        if hasattr(self.active_inference, 'compute_contrastive_loss'):
            contrastive_loss = self.active_inference.compute_contrastive_loss(
                obs, next_obs, actions
            )
            
            # Update encoder with contrastive loss
            encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.config.learning_rate
            )
            encoder_optimizer.zero_grad()
            contrastive_loss.backward()
            encoder_optimizer.step()
            
            metrics['contrastive_loss'] = contrastive_loss.item()
            
        return metrics


class PixelActiveInference(ActiveInferenceCore):
    """Active Inference for pixel observations"""
    
    def __init__(
        self,
        encoder: nn.Module,
        augmentation: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.augmentation = augmentation
        
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode pixel observation"""
        # Apply augmentation if training
        if self.augmentation is not None and self.training:
            state = self.augmentation(state)
            
        return self.encoder(state)
        
    def compute_contrastive_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss for pixel representations"""
        # Encode with augmentation
        z = self.encode_state(obs)
        z_next = self.encode_state(next_obs)
        
        # Predict next state
        z_next_pred = self.dynamics_model(z, actions)
        
        # Normalize
        z_next = F.normalize(z_next, dim=-1)
        z_next_pred = F.normalize(z_next_pred, dim=-1)
        
        # InfoNCE loss
        logits = torch.matmul(z_next_pred, z_next.T) / 0.1
        labels = torch.arange(obs.shape[0], device=obs.device)
        
        return F.cross_entropy(logits, labels)
