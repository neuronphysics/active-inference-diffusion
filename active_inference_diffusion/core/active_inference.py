"""
Core Active Inference implementation with Diffusion Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math

from .belief_dynamics import BeliefDynamics
from .diffusion import DiffusionProcess
from .free_energy import FreeEnergyComputation


class ActiveInferenceCore(nn.Module):
    """
    Core Active Inference implementation combining:
    - Variational Free Energy minimization
    - Expected Free Energy for action selection
    - Diffusion models for generative modeling
    - Fokker-Planck belief dynamics
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        config: 'ActiveInferenceConfig'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.config = config
        
        # Initialize components
        self._build_models()
        
        # Belief dynamics
        self.belief_dynamics = BeliefDynamics(
            latent_dim=latent_dim,
            config=config.belief_dynamics
        )
        
        # Diffusion process
        self.diffusion = DiffusionProcess(config.diffusion)
        
        # Free energy computation
        self.free_energy = FreeEnergyComputation(
            precision_init=config.precision_init
        )
        
        # Time tracking
        self.current_time = 0.0
        
    def _build_models(self):
        """Build neural network models"""
        # Score network: s_θ(z,t,π) = ∇_z log p_t(z|π)
        from ..models.score_networks import ScoreNetwork
        self.score_network = ScoreNetwork(
            state_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            time_embed_dim=self.config.hidden_dim
        )
        
        # Policy network: π(a|s)
        from ..models.policy_networks import GaussianPolicy
        self.policy_network = GaussianPolicy(
            state_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        # Value network: V(s,t)
        from ..models.value_networks import ValueNetwork
        self.value_network = ValueNetwork(
            state_dim=self.latent_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        # Dynamics model: f(s,a) -> s'
        from ..models.dynamics_models import LatentDynamicsModel
        self.dynamics_model = LatentDynamicsModel(
            state_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        # Reward predictor: r(s,a)
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        )
        
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to latent representation
        Override in subclasses for different observation types
        """
        return state
        
    def update_belief(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update belief distribution using Fokker-Planck dynamics
        
        dq(z|π)/dt = ∇·(D∇q) + ∇·(qF_π)
        """
        # Encode observation
        z_obs = self.encode_state(observation)
        
        # Get current belief
        belief_mean, belief_cov = self.belief_dynamics.get_parameters()
        
        # Compute score at belief mean
        t = torch.tensor([self.current_time], device=observation.device)
        
        if action is None:
            # Use policy to get action if not provided
            with torch.no_grad():
                action_dist = self.policy_network(belief_mean.unsqueeze(0))
                action = action_dist.mean
        
        score = self.score_network(
            belief_mean.unsqueeze(0),
            t,
            action
        ).squeeze(0)
        
        # Update belief
        new_mean, new_cov = self.belief_dynamics.update(
            observation=z_obs,
            score_function=score,
            action=action
        )
        
        # Update time
        self.current_time += self.config.belief_dynamics.dt
        
        return {
            'belief_mean': new_mean,
            'belief_covariance': new_cov,
            'belief_entropy': self.belief_dynamics.entropy(),
            'observation_latent': z_obs
        }
        
    def compute_expected_free_energy(
        self,
        state: torch.Tensor,
        horizon: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute expected free energy G(π) for action selection
        
        G(π) = E_q[D_KL[q(o_τ|π)||p(o_τ)] + E_q[H[p(o_τ|s_τ)]]]
             = Risk + Ambiguity
             = -Extrinsic Value + Epistemic Value
        """
        if horizon is None:
            horizon = self.config.expected_free_energy_horizon
            
        batch_size = state.shape[0]
        device = state.device
        
        # Initialize
        total_efe = torch.zeros(batch_size, device=device)
        epistemic_values = []
        extrinsic_values = []
        
        current_state = state
        discount = 1.0
        
        for step in range(horizon):
            # Sample action from policy
            action_dist = self.policy_network(current_state)
            action = action_dist.rsample()
            
            # Predict next state
            next_state = self.dynamics_model(current_state, action)
            
            # Epistemic value (information gain)
            epistemic = self._compute_epistemic_value(
                current_state, next_state, action
            )
            
            # Extrinsic value (expected reward)
            extrinsic = self._compute_extrinsic_value(
                current_state, action, next_state
            )
            
            # Accumulate EFE
            step_efe = (
                self.config.epistemic_weight * epistemic -
                self.config.extrinsic_weight * extrinsic
            )
            
            total_efe += discount * step_efe
            
            # Store for logging
            epistemic_values.append(epistemic)
            extrinsic_values.append(extrinsic)
            
            # Update state and discount
            current_state = next_state
            discount *= self.config.discount_factor
            
        info = {
            'epistemic_values': torch.stack(epistemic_values),
            'extrinsic_values': torch.stack(extrinsic_values),
            'horizon': horizon
        }
        
        return total_efe, info
        
    def _compute_epistemic_value(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute epistemic value (information gain)
        Approximated by uncertainty reduction
        """
        # Use score network to estimate uncertainty
        t = torch.zeros(state.shape[0], device=state.device)
        
        # Score variance as proxy for uncertainty
        with torch.enable_grad():
            state_var = state.requires_grad_(True)
            score = self.score_network(state_var, t, action)
            
            # Compute score divergence (Laplacian of log p)
            score_div = torch.zeros(state.shape[0], device=state.device)
            for i in range(self.latent_dim):
                grad = torch.autograd.grad(
                    score[:, i].sum(), state_var,
                    create_graph=True, retain_graph=True
                )[0]
                score_div += grad[:, i]
                
        # Higher divergence = higher uncertainty = higher epistemic value
        epistemic_value = torch.clamp(score_div, min=0.0)
        
        return epistemic_value
        
    def _compute_extrinsic_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute extrinsic value (expected reward + future value)
        """
        # Predict immediate reward
        state_action = torch.cat([state, action], dim=-1)
        reward = self.reward_predictor(state_action).squeeze(-1)
        
        # Predict future value
        t = torch.zeros(next_state.shape[0], device=next_state.device)
        future_value = self.value_network(next_state, t).squeeze(-1)
        
        # Total extrinsic value
        extrinsic_value = reward + self.config.discount_factor * future_value
        
        return extrinsic_value
        
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action based on active inference principles
        """
        # Update belief with observation
        belief_info = self.update_belief(observation)
        
        # Use belief mean for action selection
        state = belief_info['belief_mean'].unsqueeze(0)
        
        # Compute expected free energy
        efe, efe_info = self.compute_expected_free_energy(state)
        
        # Select action using policy
        action_dist = self.policy_network(state)
        
        if deterministic:
            action = action_dist.mean
        else:
            action = action_dist.rsample()
            
        # Bound action
        action = torch.tanh(action)
        
        # Prepare info
        info = {
            'expected_free_energy': efe.item(),
            'belief_entropy': belief_info['belief_entropy'].item(),
            'action_mean': action_dist.mean,
            'action_std': action_dist.stddev,
            **belief_info,
            **efe_info
        }
        
        return action, info
        
    def compute_free_energy_loss(
        self,
        states: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute variational free energy loss for training
        
        F = E_q[log q(z) - log p(z,o)]
          = D_KL[q(z)||p(z)] - E_q[log p(o|z)]
          = Complexity - Accuracy
        """
        return self.free_energy.compute_loss(
            states, observations, actions,
            score_network=self.score_network,
            current_time=self.current_time
        )
        
    def train_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Train dynamics model and reward predictor"""
        # Dynamics prediction loss
        predicted_next_states = self.dynamics_model(states, actions)
        dynamics_loss = F.mse_loss(predicted_next_states, next_states)
        
        # Reward prediction loss
        state_action = torch.cat([states, actions], dim=-1)
        predicted_rewards = self.reward_predictor(state_action).squeeze(-1)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        
        total_loss = dynamics_loss + reward_loss
        
        return {
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'total_loss': total_loss.item()
        }
        
    def update_value_function(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update value function using TD learning"""
        t = torch.zeros(states.shape[0], device=states.device)
        
        # Current values
        current_values = self.value_network(states, t).squeeze(-1)
        
        # Target values
        with torch.no_grad():
            next_values = self.value_network(next_states, t).squeeze(-1)
            target_values = rewards + self.config.discount_factor * (1 - dones) * next_values
            
        # TD loss
        value_loss = F.mse_loss(current_values, target_values)
        
        return {
            'value_loss': value_loss.item(),
            'mean_value': current_values.mean().item(),
            'mean_target': target_values.mean().item()
        }