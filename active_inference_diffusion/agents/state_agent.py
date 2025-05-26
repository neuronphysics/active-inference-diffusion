import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional

from .base_agent import BaseActiveInferenceAgent
from ..core.active_inference import ActiveInferenceCore
from ..encoder import EncoderFactory
from ..utils.buffers import ReplayBuffer
from ..configs.config import ActiveInferenceConfig, TrainingConfig


class StateBasedAgent(BaseActiveInferenceAgent):
    """Agent for state-based observations"""
    
    def _setup_dimensions(self):
        """Setup dimensions from environment"""
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Get dimensions
        if isinstance(self.obs_space, gym.spaces.Box):
            self.state_dim = self.obs_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {type(self.obs_space)}")
            
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
            
        # Update config
        self.config.state_dim = self.state_dim
        self.config.action_dim = self.action_dim
        total_input_dim = self.state_dim + self.action_dim
        print(f"dynamic input dimension: {total_input_dim}")
        
    def _build_models(self):
        """Build models for state-based observations"""
        # State encoder (can be identity or MLP)
        self.encoder = EncoderFactory.create_encoder(
            encoder_type='state',
            obs_shape=(self.state_dim,),
            feature_dim=self.config.latent_dim,
            use_projection=self.state_dim != self.config.latent_dim
        )
        
        # Active inference core
        self.active_inference = StateActiveInference(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            config=self.config,
            encoder=self.encoder
        )
        
        # Move to device
        self.active_inference = self.active_inference.to(self.device)
        
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer for states"""
        return ReplayBuffer(
            capacity=self.training_config.buffer_size,
            obs_shape=(self.state_dim,),
            action_dim=self.action_dim,
            device=self.device
        )
        
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert state observation to tensor"""
        return torch.FloatTensor(observation).unsqueeze(0)
        
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of state observations"""
        return observations.to(self.device)

    
class StateActiveInference(ActiveInferenceCore):
    """Active Inference for state observations with proper encoding"""
    
    def __init__(self, encoder: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state using provided encoder with dimension validation"""
        # Apply encoder
        encoded = self.encoder(state)
        
        # Validate dimensions
        expected_dim = self.latent_dim
        actual_dim = encoded.shape[-1]
        
        if actual_dim != expected_dim:
            raise ValueError(f"Encoding dimension mismatch: expected {expected_dim}, got {actual_dim}")
            
        return encoded

