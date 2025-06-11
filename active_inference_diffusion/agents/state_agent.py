"""
Enhanced State-Based Agent with Diffusion-Generated Latents
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any

from .base_agent import BaseActiveInferenceAgent
from ..core.active_inference import DiffusionActiveInference
from ..utils.buffers import ReplayBuffer
from ..configs.config import ActiveInferenceConfig, TrainingConfig
import torch.nn.functional as F

class DiffusionStateAgent(BaseActiveInferenceAgent):
    """
    State-based agent using diffusion-generated latent active inference
    Optimized for MuJoCo continuous control
    """
    
    def _setup_dimensions(self):
        """Setup dimensions from MuJoCo environment"""
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        if isinstance(self.obs_space, gym.spaces.Box):
            self.observation_dim = self.obs_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {type(self.obs_space)}")
            
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
            
        # Update config
        self.config.observation_dim = self.observation_dim
        self.config.action_dim = self.action_dim
        
    def _build_models(self):
        """Build diffusion active inference models"""
        # Core diffusion active inference
        self.active_inference = DiffusionActiveInference(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            config=self.config
        )
        
        # Move to device
        self.active_inference = self.active_inference.to(self.device)
        
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer for state observations"""
        return ReplayBuffer(
            capacity=self.training_config.buffer_size,
            obs_shape=(self.observation_dim,),
            action_dim=self.action_dim,
            device=self.device
        )
        
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert state observation to tensor"""
        return torch.FloatTensor(observation).unsqueeze(0)
        
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of state observations"""
        return observations.to(self.device)
        
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
        #TODO:Please check whether this works for normal observation
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
        
    def train_step(self) -> Dict[str, float]:
        """Enhanced training step for diffusion active inference"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
            
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        observations = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_observations = batch['next_observations'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        metrics = {}
        # First, update reward normalizer statistics
        self.reward_normalizer.update(rewards.cpu().numpy())

        # Normalize rewards
        normalized_rewards = torch.tensor(
                self.reward_normalizer.normalize(rewards.cpu().numpy()),
                device=self.device,
                dtype=torch.float32
        )
        # 1. Generate latents via diffusion (no gradients needed here)
        with torch.no_grad():
            belief_info = self.active_inference.update_belief_via_diffusion(observations)
            latents = belief_info['latent']
            
            next_belief_info = self.active_inference.update_belief_via_diffusion(next_observations)
            next_latents = next_belief_info['latent']
        
        # 2. Train diffusion components
        torch.nn.utils.clip_grad_norm_(
            self.active_inference.latent_score_network.parameters(),
            0.1
        )  # Clip gradients of score network
        self.score_optimizer.zero_grad()
        elbo_loss, elbo_info = self.active_inference.compute_diffusion_elbo(
            observations, normalized_rewards, latents
        )
        elbo_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.active_inference.latent_score_network.parameters()) +
            list(self.active_inference.latent_diffusion.parameters()),
            self.config.gradient_clip
        )
        self.score_optimizer.step()
        self.score_ema.update()
        metrics.update(elbo_info)
        
        # 3. Train policy network
        self.policy_optimizer.zero_grad()
        
        # Compute expected free energy
        efe, efe_info = self.active_inference.compute_expected_free_energy_diffusion(
            latents,
            horizon=self.config.efe_horizon
        )
        
        policy_loss = efe.mean()
        policy_loss.backward()
        
        nn.utils.clip_grad_norm_(
            self.active_inference.policy_network.parameters(),
            self.config.gradient_clip
        )
        self.policy_optimizer.step()
        
        metrics['policy_loss'] = policy_loss.item()
        metrics.update(efe_info)
        
        # 4. Train value network
        self.value_optimizer.zero_grad()
        
        # Predict values
        batch_size = latents.shape[0]
        time_current = torch.zeros(batch_size, device=self.device)
        time_next = torch.ones(batch_size, device=self.device)  # Next timestep
    
        # Predict values with time conditioning
        values = self.active_inference.value_network(latents, time_current).squeeze(-1)

        # Compute targets with TD learning
        with torch.no_grad():
            next_belief_info = self.active_inference.update_belief_via_diffusion(next_observations)
            next_latents = next_belief_info['latent']
            next_values = self.active_inference.value_network(next_latents,time_next).squeeze(-1)
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
        
        nn.utils.clip_grad_norm_(
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

        # 5. Train dynamics model
        self.dynamics_optimizer.zero_grad()
        
        # Predict next latents
        predicted_next_latents, predict_next_logvar = self.active_inference.predict_next_latent(latents, actions)
        
        # Dynamics loss
        dynamics_loss = F.mse_loss(predicted_next_latents, next_latents)
        dynamics_loss.backward()
        
        nn.utils.clip_grad_norm_(
            self.active_inference.latent_dynamics.parameters(),
            self.config.gradient_clip
        )
        self.dynamics_optimizer.step()
        
        metrics['dynamics_loss'] = dynamics_loss.item()
        self.total_steps += 1
        
        return metrics
        
    def _setup_optimizers(self):
        """Setup optimizers for all components"""
        # Score network optimizer (includes diffusion components)
        self.score_optimizer = torch.optim.AdamW(
            list(self.active_inference.latent_score_network.parameters()) +
            list(self.active_inference.latent_diffusion.parameters()),
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