"""
Active Inference with Diffusion-Generated Latent Spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from ..configs import ActiveInferenceConfig
from ..encoder.visual_encoders import ConvDecoder
from .diffusion import LatentDiffusionProcess
from ..models.score_networks import LatentScoreNetwork
from ..models.policy_networks import DiffusionConditionedPolicy
from ..models.value_networks import ValueNetwork
from ..models.dynamics_models import LatentDynamicsModel
class DiffusionActiveInference(nn.Module):
    """
    Core Active Inference implementation with diffusion-generated latents
    
    Key innovations:
    - Latent beliefs emerge from reverse diffusion process
    - Policies conditioned on continuous latent manifolds
    - Expected Free Energy computed over diffusion trajectories
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        config: 'ActiveInferenceConfig'
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.config = config
        self.epistemic_dropout_rate = 0.2 
        # Initialize components
        self._build_models()
        
        # Current belief state (diffusion-generated)
        self.current_latent = None
        self.latent_trajectory = []
        
    def _build_models(self):
        """Build core models for diffusion active inference"""
   
        # Latent diffusion process
        self.latent_diffusion = LatentDiffusionProcess(self.config.diffusion, latent_dim=self.latent_dim)
        # Add reward preference components
        self.register_buffer('reward_mean', torch.tensor(0.0))
        self.register_buffer('reward_var', torch.tensor(1.0))
        self.register_buffer('preference_temperature', torch.tensor(self.config.preference_temperature))
    

        # Score network for latent generation
        
        self.latent_score_network = LatentScoreNetwork(
            latent_dim=self.latent_dim,
            observation_dim=self.observation_dim,
            hidden_dim=self.config.hidden_dim,
            use_attention=True
        )
        
        # Policy network conditioned on diffusion latents
        
        self.policy_network = DiffusionConditionedPolicy(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            use_state_dependent_std=True
        )
        
        # Value network for latent states
        
        self.value_network = ValueNetwork(
            state_dim=self.latent_dim,  # Using latent dimension as state dimension
            hidden_dim=self.config.hidden_dim,
            time_embed_dim=128,  # Time embedding dimension
            num_layers=3
     )
        
        # Dynamics model in latent space
        
        self.latent_dynamics = LatentDynamicsModel(
            state_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=3
        )
        
        # Observation decoder (latent -> observation prediction)
        if not self.config.pixel_observation:
           self.observation_decoder = nn.ModuleList([
                nn.Sequential(
                nn.Linear(self.latent_dim, self.config.hidden_dim * 2),
                nn.LayerNorm(self.config.hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(self.epistemic_dropout_rate),
                ),
                nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2),
                nn.LayerNorm(self.config.hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(self.epistemic_dropout_rate),
               ),
                nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.epistemic_dropout_rate),
               ),
               nn.Linear(self.config.hidden_dim, self.observation_dim)
               ])
        else:
            self.observation_decoder = ConvDecoder(
                latent_dim=self.latent_dim,
                output_dim=self.observation_dim,
                hidden_dim=self.config.hidden_dim,
                num_conv_layers=3
                )
        #initialize a reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 2)
        )

    def decode_observation(self, latent: torch.Tensor) -> torch.Tensor:
        """Enhanced decoding with residual connections"""
        if self.config.pixel_observation:
            # For pixel observations, use convolutional decoder
            return self.observation_decoder(latent)
        else: #For non-pixel observations, use fully connected decoder
            h = latent
            # First layer
            h1 = self.observation_decoder[0](h)
            # Second layer with skip
            h2 = self.observation_decoder[1](h1)
            h2 = h2 + h1  # Skip connection
            # Third layer
            h3 = self.observation_decoder[2](h2)
            # Output layer
            return self.observation_decoder[3](h3)

    def predict_reward_from_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use existing reward predictor to get reward distribution from latent
        """
        reward_params = self.reward_predictor(latent)
        reward_mean = reward_params[:, 0]
        reward_std = torch.exp(torch.clamp(reward_params[:, 1], min=-5, max=2))
        return reward_mean, reward_std
    

  

    def update_belief_via_diffusion(
        self,
        observation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Update belief using reverse diffusion process
        This is the core innovation - beliefs as diffusion-generated latents
        """
        batch_size = observation.shape[0] if observation.dim() > 1 else 1
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        # Generate latent via reverse diffusion conditioned on observation
        trajectory = self.latent_diffusion.generate_latent_trajectory(
            score_network=self.latent_score_network,
            batch_size=batch_size,
            observation=observation,
            deterministic=False
        )
        
        # Final latent is the belief
        self.current_latent = trajectory[-1]
        self.latent_trajectory = trajectory
        
        # Compute latent statistics
        latent_mean = self.current_latent.mean(dim=0)
        latent_std = self.current_latent.std(dim=0)
        
        # Decode to observation space for validation
        predicted_obs = self.decode_observation(self.current_latent)
        reconstruction_error = F.mse_loss(predicted_obs, observation)
        
        return {
            'latent': self.current_latent,
            'latent_mean': latent_mean,
            'latent_std': latent_std,
            'trajectory_length': len(trajectory),
            'reconstruction_error': reconstruction_error,
            'observation': observation
        }
        
    def compute_expected_free_energy_diffusion(
        self,
        latent: torch.Tensor,
        horizon: int = 5,
        num_trajectories: int = 15,
        num_ambiguity_samples: int = 10
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expected Free Energy over diffusion-generated trajectories
        G(π) = Epistemic + Pragmatic + Consistency terms
        G(π) = E_z~p_θ(z) E_q(s'|s,π) [D_KL[q(o'|s')||p(o')] - log p_φ(π|z)]
        """
        device = latent.device
        batch_size = latent.shape[0]
        
        # Initialize accumulators
        total_efe = torch.zeros(batch_size, device=device)
        epistemic_values = []
        pragmatic_values = []
        latent_consistency = []
        
        # Generate future latent trajectories
        for traj_idx in range(num_trajectories):
            current_latent = latent.clone()
            traj_efe = 0
            
            for t in range(horizon):
                # Sample policy from current latent
                action, log_prob, policy_dist = self.policy_network(current_latent)
                
                # Predict next latent
                next_latent_mean, next_latent_logvar = self.predict_next_latent(current_latent, action)
                next_latent = self.reparameterize(next_latent_mean, next_latent_logvar)
                # 1. Pragmatic value (reward prediction)
                # For p(o) ∝ exp(r(o)/τ), we have ln p(o) = r(o)/τ - ln Z
                # Since Z is constant across policies, we can ignore it
                predicted_reward_mean, _ = self.predict_reward_from_latent(next_latent)
                # This makes high-reward states preferred under EFE p(o) ∝ exp(r(o)/τ)
                pragmatic = self.config.pragmatic_weight * (predicted_reward_mean / self.preference_temperature)
                 # Pragmatic value: Expected value under policy
                time_tensor = torch.full((batch_size,), float(t), device=device)
                value = self.value_network(next_latent, time_tensor).squeeze(-1)
                pragmatic += value
                # 2. Consistency (negative policy entropy)-> exploration bonus
                consistency = -policy_dist.entropy().sum(dim=-1)

    
                epistemic= self.compute_epistemic_value(
                    next_latent_mean,
                    next_latent_logvar,
                    num_samples=num_ambiguity_samples
                )
                
                # Accumulate EFE
                step_efe = (
                    self.config.epistemic_weight * epistemic +
                    self.config.pragmatic_weight * pragmatic +
                    self.config.consistency_weight * consistency
                )
                
                traj_efe += (self.config.discount_factor ** t) * step_efe
                
                # Update for next step
                current_latent = next_latent
                
            total_efe += traj_efe / num_trajectories
            
            # Store components for analysis
            epistemic_values.append(epistemic)
            pragmatic_values.append(pragmatic)
            latent_consistency.append(consistency)
            
        info = {
            'epistemic_mean': torch.stack(epistemic_values).mean(),
            'pragmatic_mean': torch.stack(pragmatic_values).mean(),
            'consistency_mean': torch.stack(latent_consistency).mean(),
            'num_trajectories': num_trajectories,
            'horizon': horizon
        }
        
        return total_efe, info
    def compute_epistemic_value(
        self,   
        next_latent_mean: torch.Tensor,
        next_latent_logvar: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        # Compute epistemic value: H(o|s,π) - H(o|s,θ,π)
        # Epistemic value (ambiguity - observation uncertainty)
        #- H[p(o|s,π)] is entropy marginalizing over model parameters (using dropout)
        #- H[p(o|s,θ,π)] is entropy for a fixed set of parameters

        batch_size = next_latent_mean.shape[0]
        device = next_latent_mean.device
        
                
        # Enable dropout for sampling
        original_mode = self.training
        entropy_obs_given_state_policy= torch.zeros(batch_size, device=device)
        self.train()
        observation=[]
        # Term 2.1: Sample different parameters (dropout)
        for _ in range(num_samples):
            # Different parameters each forward pass
            next_latent= self.reparameterize(next_latent_mean, next_latent_logvar)
            po_temp = self.decode_observation(next_latent) 
            observation.append(po_temp)
        obs_dropout = torch.stack(observation, dim=0)  # Shape: (num_samples, batch_size, obs_dim)
        if self.config.pixel_observation:
            
            # Compute entropy (assuming Bernoulli distribution)
            # Shape: (batch_size, obs_dim)
            p = torch.clamp(obs_dropout, 1e-8, 1-1e-8)
            entropy = - (p * torch.log(p) + (1-p) * torch.log(1-p))
            entropy_obs_given_state_policy = entropy.mean(0).sum(dim=(1,2,3))
        else:
            var_obs = obs_dropout.var( dim=0)+ 1e-8  # Add small constant for numerical stability
    
            # Differential entropy of Gaussian: 0.5 * log(2πe * σ²)
            entropy = 0.5 * torch.log(2 * np.pi * np.e * var_obs + 1e-8)
    
            entropy_obs_given_state_policy=entropy.sum(dim=-1) 
        entropy_obs_given_state_theta_pi = torch.zeros(batch_size, device=device)    
        # Sample different states (same parameters)
        self.eval()
        observation = []
        with torch.no_grad():
            for _ in range(num_samples):
                # Sample from state distribution
                state_sample = self.reparameterize(next_latent_mean, next_latent_logvar)
                po_temp = self.decode_observation(state_sample)
                observation.append(po_temp)
            po_temp = torch.stack(observation, dim=0)  # Shape: (num_samples, batch_size, obs_dim)
            if self.config.pixel_observation:
                # Shape: (batch_size, obs_dim)
                p = torch.clamp(po_temp, 1e-8, 1-1e-8)
                entropy = - (p * torch.log(p) + (1-p) * torch.log(1-p))
                entropy_obs_given_state_theta_pi= entropy.mean(0).sum(dim=(1,2,3))
            else:
                state_var = torch.var(po_temp, dim=0)
                # Differential entropy of Gaussian: 0.5 * log(2πe * σ²)
                entropy = 0.5 * torch.log(2 * np.pi * np.e * state_var + 1e-8)
                entropy_obs_given_state_theta_pi= entropy.sum(dim=-1) 
            
                self.train(original_mode)  # Restore original training mode
                # Epistemic value: H(o|s,π) - H(o|s,θ,π)
                epistemic_value=entropy_obs_given_state_policy - entropy_obs_given_state_theta_pi
                epistemic_value = torch.clamp(epistemic_value, min=0.0)  # Ensure non-negative
                return epistemic_value

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

        
    def predict_next_latent(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent state using learned dynamics"""
        delta = self.latent_dynamics(latent, action)
        next_mean= latent + delta  # Residual connection
        next_logvar = torch.full_like(next_mean, np.log(0.1))
        return next_mean, next_logvar
        
    def _compute_latent_kl(
        self,
        latent: torch.Tensor,
        prior_latent: torch.Tensor
        ) -> torch.Tensor:
        """KL divergence between latent distributions"""
        # Assume Gaussian with unit variance for simplicity
        
        # Can be extended to learned variances
        kl = 0.5 * torch.sum((latent - prior_latent) ** 2, dim=-1)
        return kl
        
    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action using diffusion-generated latent active inference
        """
        # Update belief via diffusion
        belief_info = self.update_belief_via_diffusion(observation)
        
        # Current latent belief
        latent = belief_info['latent']
        
        # Compute expected free energy
        efe, efe_info = self.compute_expected_free_energy_diffusion(
            latent,
            horizon=self.config.efe_horizon
        )
        
        # Get action from policy conditioned on latent
        action, log_prob, policy_dist = self.policy_network(
            latent,
            deterministic=deterministic
        )
        
        # Compile information
        info = {
            **belief_info,
            'expected_free_energy': efe.mean().item(),
            'action_log_prob': log_prob.mean().item(),
            'policy_entropy': policy_dist.entropy().sum(dim=-1).mean().item(),
            **efe_info
        }
        
        return action, info
        
    def compute_diffusion_elbo(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        latents: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Annealing time conditioned ELBO for diffusion-generated latents
        Modified ELBO for diffusion-generated latent active inference
        
        L = E_q(z|o,π)[log p(o|z)] - D_KL[q(z|o,π)||p_θ(z)] + R_diffusion(θ)
        """
        batch_size = observations.shape[0]
        device = observations.device
        
        # Generate latents if not provided
        if latents is None:
            # Use current belief generation
            belief_info = self.update_belief_via_diffusion(observations)
            latents = belief_info['latent']

                   
        # Reconstruction term
        predicted_obs = self.decode_observation(latents)
        reconstruction_loss = F.mse_loss(predicted_obs, observations)
        
                
        # Diffusion score matching loss
        # Sample continuous time with importance sampling
        # Emphasize times where loss is typically high
        if hasattr(self, 'time_importance_weights'):
            # Use learned importance weights
            t = self._importance_sample_time(batch_size, device)
        else:
            # Uniform sampling initially
            t = torch.rand(batch_size, device=device)

        
        noise = torch.randn_like(latents)
    
        noisy_latents, true_noise, sample_info = self.latent_diffusion.continuous_q_sample(latents, t, noise)
        
        predicted_score = self.latent_score_network(
            noisy_latents,
            t,
            observations
        )
        # Compute true score with proper scaling
        log_snr = sample_info['log_snr']
        alpha = sample_info['alpha']
        sigma = sample_info['sigma']
    
        # True score: -noise / sigma (not sqrt(1-alpha) for continuous time)
        true_score = -noise / (sigma + 1e-8)
    
        # Annealed loss weight
        loss_weight = self.latent_diffusion.compute_loss_weight(t)
        # Score matching loss with annealing
        score_diff = predicted_score - true_score
        
        per_sample_losses = loss_weight.view(-1, 1) * torch.sum(score_diff ** 2, dim=-1)
        score_matching_loss = torch.mean(per_sample_losses)

        # Add gradient penalty for stability
        grad_penalty = self._compute_gradient_penalty(noisy_latents, t, observations)
    
        # KL term with annealing
        prior_latents = self.latent_diffusion.sample_latent_prior(batch_size, device)
        kl_loss = self._compute_latent_kl(latents, prior_latents).mean()
        kl_weight = torch.exp(-5.0 * t.mean())  # Anneal KL over time

        # Add reward prediction loss if rewards provided
        predicted_rewards = self.reward_predictor(latents)
        rewards_mean =predicted_rewards[:, 0]
        rewards_std = torch.exp(torch.clamp(predicted_rewards[:, 1],min=-5, max=2))
        rewards_distribution = torch.distributions.Normal(rewards_mean, rewards_std)
        reward_loss = -rewards_distribution.log_prob(rewards).mean()
        # Total ELBO
        elbo = -reconstruction_loss + self.config.kl_weight * kl_loss*kl_weight + \
               self.config.diffusion_weight * score_matching_loss+ \
                0.1*grad_penalty - \
               self.config.reward_weight * reward_loss
        self._update_time_importance(t, per_sample_losses.detach())               
        info = {
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'score_matching_loss': score_matching_loss.item(),
            'elbo': elbo.item(),
            'reward_loss': reward_loss.item(),
            'grad_penalty': grad_penalty.item(),
            'mean_time': t.mean().item(),
            'loss_weight_mean': loss_weight.mean().item()
        }
        
        return -elbo, info  # Return negative ELBO as loss

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
        lambda_: float = 0.95,
        n_steps: int = 5
    ) -> torch.Tensor:
        """
        Compute λ-returns as in Dreamer v2.
    
        The λ-return is a weighted average of n-step returns:
        G^λ_t = (1-λ) Σ_{n=1}^{N-1} λ^{n-1} G^n_t + λ^{N-1} G^N_t
    
        Where G^n_t is the n-step return.
        """
        batch_size = rewards.shape[0]
        device = rewards.device
    
        # Initialize returns storage
        lambda_returns = torch.zeros_like(rewards)
    
        # Compute n-step returns
        for idx in range(batch_size):
            returns = []
        
            # Calculate different n-step returns
            for n in range(1, min(n_steps + 1, batch_size - idx)):
                n_step_return = 0
                discount = 1.0
            
                # Sum discounted rewards for n steps
                for k in range(n):
                    if idx + k < batch_size:
                        n_step_return += discount * rewards[idx + k]
                        discount *= self.config.discount_factor * (1 - dones[idx + k].float())
                    
                # Add bootstrapped value
                if idx + n < batch_size and not dones[idx + n - 1]:
                    n_step_return += discount * next_values[idx + n]
                
                returns.append(n_step_return)
        
            # Compute weighted average with λ
            if returns:
                weighted_return = 0
                lambda_sum = 0
            
                for i, ret in enumerate(returns[:-1]):
                    weight = (1 - lambda_) * (lambda_ ** i)
                    weighted_return += weight * ret
                    lambda_sum += weight
                
                # Last return gets remaining weight
                if len(returns) > 0:
                    last_weight = lambda_ ** (len(returns) - 1)
                    weighted_return += last_weight * returns[-1]
                    lambda_sum += last_weight
                
                lambda_returns[idx] = weighted_return / (lambda_sum + 1e-8)
            else:
                lambda_returns[idx] = rewards[idx] + self.config.discount_factor * (1 - dones[idx].float()) * next_values[idx]
    
        return lambda_returns
      
    def _compute_gradient_penalty(
        self,
        noisy_latents: torch.Tensor,
        t: torch.Tensor,
        observations: torch.Tensor
    ) -> torch.Tensor:
        """Gradient penalty for stable training"""
        noisy_latents.requires_grad_(True)
    
        score = self.latent_score_network(noisy_latents, t, observations)
    
        gradients = torch.autograd.grad(
            outputs=score.sum(),
            inputs=noisy_latents,
            create_graph=True,
            retain_graph=True
        )[0]
    
        grad_norm = gradients.norm(2, dim=1)
        penalty = torch.mean((grad_norm - 1.0) ** 2)
    
        return penalty

    def _importance_sample_time(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Sample time with importance weights based on loss history"""
        if not hasattr(self, 'time_importance_weights'):
            # Initialize uniform
            self.time_importance_weights = torch.ones(100, device=device)
    
        # Sample from categorical distribution
        probs = F.softmax(self.time_importance_weights, dim=0)
        indices = torch.multinomial(probs, batch_size, replacement=True)
    
        # Convert to continuous time
        t = (indices.float() + torch.rand(batch_size, device=device)) / 100.0
    
        return t

    def _update_time_importance(
        self,
        t: torch.Tensor,
        loss: torch.Tensor
    ):
        """Update importance weights for time sampling"""
        if not hasattr(self, 'time_importance_weights'):
            self.time_importance_weights = torch.ones(100, device=t.device)
    
        # Discretize time
        indices = (t * 99).long().clamp(0, 99)
    
        # Update weights with EMA
        for idx in range(len(indices)):
            idx = indices[idx].item()  # Convert to scalar index
            # Get current weight as a scalar
        
            # Set the new weight
            self.time_importance_weights[idx] =(
            0.99 * self.time_importance_weights[idx].item() + 
            0.01 * loss[idx].item()
          )    

def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract coefficients helper"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
class EMAModel:
    """Exponential Moving Average of model weights for stable training"""
    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
    
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
