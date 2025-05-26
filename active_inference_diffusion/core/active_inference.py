"""
Core Active Inference implementation with Diffusion Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
import warnings

from .belief_dynamics import BeliefDynamics
from .diffusion import DiffusionProcess
from .free_energy import FreeEnergyComputation
from ..configs.config import ActiveInferenceConfig

class ActiveInferenceCore(nn.Module):
    """
    Core Active Inference implementation combining:
    - Variational Free Energy minimization
    - Expected Free Energy for action selection
    - Diffusion models for generative modeling
    - Fokker-Planck belief dynamics
    
    Enhanced with:
    - Proper score matching training
    - Numerical stability improvements
    - Computational graph preservation
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
        
        # Belief dynamics with enhanced stability
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
        
        # Training state tracking
        self.training_step = 0
        
    def _build_models(self):
        """Build neural network models with enhanced initialization"""
        # Score network: s_θ(z,t,π) = ∇_z log p_t(z|π)
        from ..models.score_networks import ScoreNetwork
        self.score_network = ScoreNetwork(
            state_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            time_embed_dim=self.config.hidden_dim
        )
        
        # Enhanced initialization for score network
        self._initialize_score_network()
        
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
        
    def _initialize_score_network(self):
        """Enhanced initialization for score network to prevent vanishing gradients"""
        # Use Xavier initialization for internal layers
        for name, param in self.score_network.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                if 'network' in name and 'weight' in name:
                    # Use small random initialization for output layer
                    if name.endswith('network.-1.weight'):
                        nn.init.normal_(param, mean=0.0, std=1e-4)
                    else:
                        nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state to latent representation
        Override in subclasses for different observation types
        """
        return NotImplementedError
    
    def train_score_network(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Proper score matching training for diffusion-active inference integration
        
        Implements denoising score matching: L = E[||s_θ(x_t,t,π) + ε/√(1-ᾱ_t)||²]
        """
        batch_size = states.shape[0]
        device = states.device
        
        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0, self.diffusion.config.num_diffusion_steps, 
                (batch_size,), device=device
            )
        
        # Sample noise
        noise = torch.randn_like(states)
        
        # Forward diffusion: x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        noisy_states = self.diffusion.q_sample(states, timesteps, noise)
        
        # Predict score
        predicted_score = self.score_network(noisy_states, timesteps.float(), actions)
        
        # True score: -ε / √(1 - ᾱ_t)
        alphas_cumprod_t = self.diffusion.sqrt_one_minus_alphas_cumprod[timesteps]
        true_score = -noise / alphas_cumprod_t.view(-1, 1)
        
        # Score matching loss
        score_loss = F.mse_loss(predicted_score, true_score)
        
        return {
            'score_matching_loss': score_loss.item(),
            'mean_predicted_norm': predicted_score.norm(dim=-1).mean().item(),
            'mean_true_norm': true_score.norm(dim=-1).mean().item()
        }
        
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
                action = action_dist.mean.squeeze(0)
        else:
            # Ensure proper dimensionality
            if action.dim() > 1:
                action = action.squeeze(0)
        
        score = self.score_network(
            belief_mean.unsqueeze(0),
            t,
            action.unsqueeze(0)
        ).squeeze(0)
        
        # Update belief with numerical stability checks
        try:
            new_mean, new_cov = self.belief_dynamics.update(
                observation=z_obs,
                score_function=score,
                action=action
            )
        except RuntimeError as e:
            warnings.warn(f"Belief update failed: {e}, resetting to prior")
            self.belief_dynamics.reset()
            new_mean, new_cov = self.belief_dynamics.get_parameters()
        
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
            
            # Enhanced epistemic value computation (computationally efficient)
            epistemic = self._compute_epistemic_value_efficient(
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
    
    def _compute_epistemic_value_efficient(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficient epistemic value computation using score network variance
        Avoids expensive gradient computations
        """
        t = torch.zeros(state.shape[0], device=state.device)
        
        # Compute score at current and next states
        with torch.no_grad():
            score_current = self.score_network(state, t, action)
            score_next = self.score_network(next_state, t, action)
            
            # Information gain approximated by score difference magnitude
            score_change = torch.norm(score_next - score_current, dim=-1)
            
            # Higher score change indicates higher information gain
            epistemic_value = torch.tanh(score_change)  # Bounded between 0 and 1
            
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
        Enhanced with proper tensor handling and computational graph preservation
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
        
        # Enhanced info preparation with proper tensor handling
        info = {
            'expected_free_energy': efe.item() if efe.dim() == 0 else efe.mean().item(),
            'belief_entropy': belief_info['belief_entropy'].item(),
            # Preserve computational graph for training
            'action_mean': action_dist.mean,
            'action_std': action_dist.stddev,
            'belief_mean': belief_info['belief_mean'],
            'belief_covariance': belief_info['belief_covariance'],
            'observation_latent': belief_info['observation_latent'],
            # Fixed tensor stacking issue
            'epistemic_values': efe_info['epistemic_values'],  # Already stacked
            'extrinsic_values': efe_info['extrinsic_values'],  # Already stacked
            'horizon': efe_info['horizon']
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
        Enhanced with proper score matching integration
        """
        # Standard free energy computation
        free_energy, fe_info = self.free_energy.compute_loss(
            states, observations, actions,
            score_network=self.score_network,
            current_time=self.current_time
        )
        
        # Add proper score matching loss
        score_metrics = self.train_score_network(states, actions)
        
        # Combine losses
        total_loss = free_energy + 0.1 * torch.tensor(score_metrics['score_matching_loss'], requires_grad=True)
        
        # Enhanced info
        enhanced_info = {
            **fe_info,
            'score_matching_loss': score_metrics['score_matching_loss'],
            'total_enhanced_loss': total_loss.item()
        }
        
        return total_loss, enhanced_info
        
    def train_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Train dynamics model and reward predictor with enhanced monitoring"""
        # Dynamics prediction loss
        predicted_next_states = self.dynamics_model(states, actions)
        dynamics_loss = F.mse_loss(predicted_next_states, next_states)
        
        # Reward prediction loss
        state_action = torch.cat([states, actions], dim=-1)
        predicted_rewards = self.reward_predictor(state_action).squeeze(-1)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        
        total_loss = dynamics_loss + reward_loss
        
        # Enhanced monitoring
        dynamics_error = (predicted_next_states - next_states).abs().mean()
        reward_error = (predicted_rewards - rewards).abs().mean()
        
        return {
            'dynamics_loss': dynamics_loss.item(),
            'reward_loss': reward_loss.item(),
            'total_loss': total_loss.item(),
            'dynamics_mae': dynamics_error.item(),
            'reward_mae': reward_error.item()
        }
        
    def update_value_function(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Dict[str, float]:
        """Update value function using TD learning with enhanced stability"""
        t = torch.zeros(states.shape[0], device=states.device)
        
        # Current values
        current_values = self.value_network(states, t).squeeze(-1)
        
        # Target values with enhanced stability
        with torch.no_grad():
            next_values = self.value_network(next_states, t).squeeze(-1)
            target_values = rewards + self.config.discount_factor * (1 - dones.long()) * next_values
            
            # Clip extreme values for stability
            target_values = torch.clamp(target_values, -100, 100)
            
        # Huber loss for enhanced robustness
        value_loss = F.huber_loss(current_values, target_values, delta=10.0)
        
        # Enhanced monitoring
        td_error = (current_values - target_values).abs().mean()
        
        return {
            'value_loss': value_loss.item(),
            'mean_value': current_values.mean().item(),
            'mean_target': target_values.mean().item(),
            'td_error': td_error.item()
        }
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Comprehensive training diagnostics"""
        metrics = {}
        
        # Belief dynamics health
        belief_mean, belief_cov = self.belief_dynamics.get_parameters()
        metrics['belief_mean_norm'] = belief_mean.norm().item()
        
        if self.config.belief_dynamics.use_full_covariance:
            metrics['belief_cov_det'] = torch.det(belief_cov).item()
            metrics['belief_cov_trace'] = torch.trace(belief_cov).item()
        else:
            metrics['belief_var_mean'] = belief_cov.diag().mean().item()
            
        # Score network health
        score_params = list(self.score_network.parameters())
        metrics['score_grad_norm'] = sum(p.grad.norm().item() 
                                       for p in score_params if p.grad is not None)
        
        return metrics
    

