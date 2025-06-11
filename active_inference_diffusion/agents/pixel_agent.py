"""
Pixel-Based Active Inference Agent
Integrates visual encoding with diffusion-generated latent spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any

from .base_agent import BaseActiveInferenceAgent
from ..core.active_inference import DiffusionActiveInference
from ..encoder.visual_encoders import RandomShiftAugmentation, DrQV2Encoder
from ..encoder.state_encoders import EncoderFactory
from ..utils.buffers import ReplayBuffer
from ..configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)


class DiffusionPixelAgent(BaseActiveInferenceAgent):
    """
    Pixel-based agent using diffusion-generated latent active inference
    
    Key innovations:
    - Visual observations encoded to feature space before latent generation
    - Contrastive learning enhances visual representation quality
    - Seamless integration with diffusion-based belief updates
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: ActiveInferenceConfig,
        training_config: TrainingConfig,
        pixel_config: PixelObservationConfig
    ):
        self.pixel_config = pixel_config
        super().__init__(env, config, training_config)
        
    def _setup_dimensions(self):
        """Setup dimensions for pixel observations"""
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Validate observation space
        if isinstance(self.obs_space, gym.spaces.Box):
            obs_shape = self.obs_space.shape
            if len(obs_shape) == 3:  # (C, H, W) or (H, W, C)
                self.frame_stack = 1
                self.obs_shape = obs_shape
                # Ensure channels-first format
                if obs_shape[-1] in [1, 3]:  # Likely (H, W, C)
                    self.obs_shape = (obs_shape[-1], obs_shape[0], obs_shape[1])
            elif len(obs_shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                self.frame_stack = obs_shape[0]
                self.obs_shape = obs_shape[1:]
                print(f"Using frame stack: {self.frame_stack}")
            else:
                raise ValueError(f"Unexpected observation shape: {obs_shape}")
        else:
            raise ValueError(f"Unsupported observation space: {type(self.obs_space)}")
            
        # Validate action space
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
            
        # Update configuration
        self.config.action_dim = self.action_dim
        # Set observation dimension to encoder output dimension
        if self.pixel_config.pixel_observation:
            self.config.observation_dim = self.config.latent_dim
        else:
            self.config.observation_dim = self.obs_space[0]
        
    def _build_models(self):
        """Build visual encoding and diffusion active inference models"""
        # Visual encoder
        self.encoder = DrQV2Encoder(
            obs_shape=self.obs_shape,
            feature_dim=self.config.latent_dim,
            frame_stack=self.pixel_config.frame_stack,
            num_layers=4,
            num_filters=32,
        )
        
        # Augmentation module
        self.augmentation = RandomShiftAugmentation(
            pad=self.pixel_config.random_shift_pad
        ) if self.pixel_config.augmentation else None
        
        # Core diffusion active inference
        # Uses encoder output dimension as observation dimension
        self.active_inference = DiffusionActiveInference(
            observation_dim=self.config.observation_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            config=self.config,
            pixel_shape=self.obs_shape if self.pixel_config.pixel_observation else None
        )
        

        # Move all components to device
        self.encoder = self.encoder.to(self.device)
        self.active_inference = self.active_inference.to(self.device)
        
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using visual encoding and diffusion active inference
        
        Pipeline: pixels -> features -> diffusion latents -> policy
        """
        # Process raw pixels
        obs_tensor = self._process_observation(observation)
        
        # Encode to feature space
        with torch.no_grad():
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder = self.encoder.to(self.device)

            encoded_obs = self.encode_observation(obs_tensor.to(self.device))
            if encoded_obs.dim() == 1:
               encoded_obs = encoded_obs.unsqueeze(0)
            
            # Use diffusion active inference with encoded features
            action_tensor, info = self.active_inference.act(
                encoded_obs.squeeze(0),  # Remove batch dimension
                deterministic=deterministic
            )
            
        # Convert to numpy
        action = action_tensor.cpu().numpy()
        # Handle different action shapes
        if action.ndim == 0:  # Scalar
            action = np.array([action])
        elif action.ndim == 2 and action.shape[0] == 1:  # [1, action_dim]
            action = action[0]
        elif action.ndim == 1:  # Already correct shape
            pass
        else:
            # Unexpected shape - try to flatten to 1D
            action = action.flatten()
        if hasattr(self, 'action_dim') and len(action) != self.action_dim:
            print(f"Warning: action shape {action.shape} doesn't match expected {self.action_dim}")
         
        # Add exploration noise if training
        if not deterministic and self.training and self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        # Add encoding info
        info['encoded_obs_norm'] = encoded_obs.norm().item()
        
        return action, info
        
    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode pixel observation to feature space with augmentation"""
        # Apply augmentation during training
        # Handle different input formats
        if observation.ndim == 5:  # (batch, frame_stack, C, H, W)
            batch_size, frame_stack, c, h, w = observation.shape
            # Reshape to (batch, frame_stack * C, H, W) for encoder
            observation = observation.view(batch_size, frame_stack * c, h, w)
        elif observation.ndim == 4 and hasattr(self, 'frame_stack') and self.frame_stack > 1:
            # Single sample with frame stack
            if observation.shape[0] == 1:  # Batch dimension
                # Check if second dimension is frame stack
                if observation.shape[1] == self.frame_stack:
                    batch_size = 1
                    observation = observation.view(batch_size, -1, 
                                             observation.shape[-2], 
                                             observation.shape[-1])
        elif observation.ndim == 3:  # Single image without batch
            observation = observation.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
    
        # Apply augmentation during training
        if self.augmentation is not None and self.training:
            observation = self.augmentation(observation)
        
        # Encode to feature space
        encoded = self.encoder(observation)
    
        # Ensure proper dimensions
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
              
        return encoded
        
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer optimized for pixel storage"""
        # Store full observation shape including frame stack
        if self.frame_stack > 1:
            buffer_obs_shape = (self.frame_stack, *self.obs_shape)
        else:
            buffer_obs_shape = self.obs_shape
        return ReplayBuffer(
            capacity=self.training_config.buffer_size,
            obs_shape=buffer_obs_shape,
            action_dim=self.action_dim,
            device=self.device,
            optimize_memory=True  # Enable compression for pixels
        )
        
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert pixel observation to tensor with proper formatting"""
        if isinstance(observation, torch.Tensor):
            obs_array = observation.detach().cpu().numpy()
        else:
            obs_array = np.array(observation)
    
        # Determine observation format and convert to channels-first
        if obs_array.ndim == 2:
            # Single grayscale image (H, W) -> (1, H, W)
            obs_array = np.expand_dims(obs_array, axis=0)
        elif obs_array.ndim == 3:
            # Check if it's (H, W, C) or (C, H, W)
            if obs_array.shape[-1] in [1, 3]:  # (H, W, C) format
                obs_array = np.transpose(obs_array, (2, 0, 1))  # -> (C, H, W)
            # else assume it's already (C, H, W)
        elif obs_array.ndim == 4:
            # Could be (frame_stack, H, W, C) or (B, C, H, W)
            if obs_array.shape[-1] in [1, 3]:  # (frame_stack, H, W, C)
                obs_array = np.transpose(obs_array, (0, 3, 1, 2))  # -> (frame_stack, C, H, W)
            # else assume it's already (B, C, H, W) or (frame_stack, C, H, W)
        else:
            raise ValueError(f"Unexpected observation shape: {obs_array.shape}")
    
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_array)
    
        # Add batch dimension if needed
        if obs_tensor.dim() == 3:  # Single image (C, H, W)
            obs_tensor = obs_tensor.unsqueeze(0)  # -> (1, C, H, W)
        elif obs_tensor.dim() == 4 and obs_tensor.shape[0] == self.frame_stack:
            # Frame stack without batch dimension (frame_stack, C, H, W)
            obs_tensor = obs_tensor.unsqueeze(0)  # -> (1, frame_stack, C, H, W)
    
        # Normalize pixel values if needed
        if obs_tensor.dtype == torch.uint8 or obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor.float() / 255.0
    
        return obs_tensor  
          
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of pixel observations"""
        # Move to device
        observations = observations.to(self.device)
        
        # Normalize to [0, 1] if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        if observations.dim() == 5 :
            batch_size, frame_stack, c, h, w = observations.shape
            if hasattr(self.encoder, 'expects_frame_stack') and self.encoder.expects_frame_stack:
                pass  # Keep as is
            else:
                # Flatten frame stack into channels
                observations = observations.view(batch_size, frame_stack * c, h, w)
        return observations
        
    def train_step(self) -> Dict[str, float]:
        """Enhanced training step with visual representation learning"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
            
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Process observations
        obs = self._process_batch_observations(batch['observations'])
        next_obs = self._process_batch_observations(batch['next_observations'])
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        metrics = {}
        
        # 1. Encode observations to feature space
        encoded_obs = self.encode_observation(obs)
        encoded_next_obs = self.encode_observation(next_obs)
        # First, update reward normalizer statistics
        self.reward_normalizer.update(rewards.cpu().numpy())

        # Normalize rewards
        normalized_rewards = torch.tensor(
                self.reward_normalizer.normalize(rewards.cpu().numpy()),
                device=self.device,
                dtype=torch.float32
                )
        # 2. Generate latents via diffusion
        with torch.no_grad():
            belief_info = self.active_inference.update_belief_via_diffusion(encoded_obs)
            latents = belief_info['latent']
            
            next_belief_info = self.active_inference.update_belief_via_diffusion(encoded_next_obs)
            next_latents = next_belief_info['latent']
        torch.nn.utils.clip_grad_norm_(self.active_inference.latent_score_network.parameters(),
                                       0.1)    
        # 3. Train diffusion components
        self.score_optimizer.zero_grad()
        elbo_loss, elbo_info = self.active_inference.compute_diffusion_elbo(
            encoded_obs, normalized_rewards, latents
        )
        
        # 4. Add contrastive representation loss
        contrastive_loss = self.compute_representation_loss(
            encoded_obs, encoded_next_obs, actions, latents, next_latents
        )
        
        # Combined loss
        total_loss = elbo_loss + self.config.contrastive_weight * contrastive_loss
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            list(self.active_inference.latent_score_network.parameters()) +
            list(self.active_inference.latent_diffusion.parameters()) +
            list(self.encoder.parameters()),
            self.config.gradient_clip
        )
        self.score_optimizer.step()
        self.score_ema.update()
        
        metrics.update(elbo_info)
        metrics['contrastive_loss'] = contrastive_loss.item()
        
        # 5. Train policy network
        self.policy_optimizer.zero_grad()
        
        efe, efe_info = self.active_inference.compute_expected_free_energy_diffusion(
            latents,
            horizon=self.config.efe_horizon
        )
        
        policy_loss = efe.mean()
        policy_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.active_inference.policy_network.parameters(),
            self.config.gradient_clip
        )
        self.policy_optimizer.step()
        
        metrics['policy_loss'] = policy_loss.item()
        metrics.update(efe_info)
        
        # 6. Train value network
        self.value_optimizer.zero_grad()

        batch_size = latents.shape[0]
        time_current = torch.zeros(batch_size, device=self.device)
        time_next = torch.ones(batch_size, device=self.device)  # Next timestep
        values = self.active_inference.value_network(latents, time_current).squeeze(-1)
        # Predict values with time conditioning
        with torch.no_grad():
            next_values = self.active_inference.value_network(next_latents, time_next).squeeze(-1)
            targets = self.active_inference.compute_lambda_returns(
                    rewards=normalized_rewards,
                    values=values,
                    next_values=next_values,
                    dones=dones,
                    lambda_=0.95,  # TODO: can be added to config
                    n_steps=5
                    )
    
        value_loss = F.huber_loss(values, targets)
        value_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.active_inference.value_network.parameters(),
            self.config.gradient_clip
        )
        self.value_optimizer.step()
        
        metrics['value_loss'] = value_loss.item()
        # Train epistemic estimator separately
        if self.total_steps % 5 == 0:  # Train less frequently for stability
            epistemic_mi, epistemic_metrics = self.active_inference.train_epistemic_estimator(
                latents, actions, next_latents
            )
            metrics['epistemic_mi'] = epistemic_mi
            metrics.update(epistemic_metrics)

        # 7. Train dynamics model
        self.dynamics_optimizer.zero_grad()
        
        predicted_next_latents, predicted_next_logvar = self.active_inference.predict_next_latent(latents, actions)
        dynamics_loss = F.mse_loss(predicted_next_latents, next_latents)
        dynamics_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.active_inference.latent_dynamics.parameters(),
            self.config.gradient_clip
        )
        self.dynamics_optimizer.step()
        
        metrics['dynamics_loss'] = dynamics_loss.item()
        self.total_steps += 1
        
        return metrics
        
    def compute_representation_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss for visual representation learning
        Ensures latent dynamics align with visual features
        """
        # Predict next visual features from current latent and action
        predicted_next_latent, predicted_logvar = self.active_inference.predict_next_latent(latents, actions)
        predicted_std = torch.exp(0.5 * predicted_logvar)
        # Normalize for contrastive loss
        pred_norm = F.normalize(predicted_next_latent, dim=-1)
        target_norm = F.normalize(next_obs, dim=-1)
        uncertainty_weights = 1.0 / (1.0 + predicted_std.mean(dim=-1, keepdim=True))        
        # InfoNCE loss
        logits = torch.matmul(pred_norm, target_norm.T) / 0.1
        weighted_logits = logits * uncertainty_weights
        labels = torch.arange(obs.shape[0], device=obs.device)

        return F.cross_entropy(weighted_logits, labels)

    def _setup_optimizers(self):
        """Setup optimizers including visual components"""
        # Score network optimizer (includes encoder)
        self.score_optimizer = torch.optim.AdamW(
            list(self.active_inference.latent_score_network.parameters()) +
            list(self.active_inference.latent_diffusion.parameters()) +
            list(self.encoder.parameters())+
            list(self.active_inference.feature_decoder.parameters()),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Policy optimizer
        self.policy_optimizer = torch.optim.AdamW(
            self.active_inference.policy_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Value optimizer
        self.value_optimizer = torch.optim.AdamW(
            self.active_inference.value_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Dynamics optimizer
        self.dynamics_optimizer = torch.optim.AdamW(
            list(self.active_inference.latent_dynamics.parameters()) +
            list(self.active_inference.observation_decoder.parameters())+
            list(self.active_inference.reward_predictor.parameters()),
            lr=self.config.learning_rate
        )
