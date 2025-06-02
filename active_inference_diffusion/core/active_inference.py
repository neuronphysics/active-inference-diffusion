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
        
        # Initialize components
        self._build_models()
        
        # Current belief state (diffusion-generated)
        self.current_latent = None
        self.latent_trajectory = []
        
    def _build_models(self):
        """Build core models for diffusion active inference"""
        
        # Latent diffusion process
        from .diffusion import LatentDiffusionProcess
        self.latent_diffusion = LatentDiffusionProcess(self.config.diffusion, latent_dim=self.latent_dim)
        
        # Score network for latent generation
        from ..models.score_networks import LatentScoreNetwork
        self.latent_score_network = LatentScoreNetwork(
            latent_dim=self.latent_dim,
            observation_dim=self.observation_dim,
            hidden_dim=self.config.hidden_dim,
            use_attention=True
        )
        
        # Policy network conditioned on diffusion latents
        from ..models.policy_networks import DiffusionConditionedPolicy
        self.policy_network = DiffusionConditionedPolicy(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            use_state_dependent_std=True
        )
        
        # Value network for latent states
        from ..models.value_networks import ValueNetwork
        self.value_network = ValueNetwork(
            state_dim=self.latent_dim,  # Using latent dimension as state dimension
            hidden_dim=self.config.hidden_dim,
            time_embed_dim=128,  # Time embedding dimension
            num_layers=3
     )
        
        # Dynamics model in latent space
        from ..models.dynamics_models import LatentDynamicsModel
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
                ),
                nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2),
                nn.LayerNorm(self.config.hidden_dim * 2),
                nn.SiLU(),
               ),
                nn.Sequential(
                nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.SiLU(),
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
        num_trajectories: int = 16
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Enhanced Expected Free Energy over diffusion-generated trajectories
        
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
                next_latent = self.predict_next_latent(current_latent, action)
                
                # Epistemic value: Information gain about latent dynamics
                latent_kl = self._compute_latent_kl(current_latent, next_latent)
                epistemic = torch.log1p(latent_kl)  # Bounded transformation
                
                # Pragmatic value: Expected value under policy
                time_tensor = torch.full((batch_size,), float(t), device=device)
                value = self.value_network(next_latent, time_tensor).squeeze(-1)
                pragmatic = value
                
                # Latent consistency: Policy entropy conditioned on latent
                consistency = -policy_dist.entropy().sum(dim=-1)
                
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
        
    def predict_next_latent(
        self,
        latent: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Predict next latent state using learned dynamics"""
        delta = self.latent_dynamics(latent, action)
        return latent + delta  # Residual connection
        
    def _compute_latent_kl(
        self,
        latent: torch.Tensor,
        prior_latent: torch.Tensor
        ) -> torch.Tensor:
        """KL divergence between latent distributions"""
        # Assume Gaussian with unit variance for simplicity
        latents_norm = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)
        prior_norm = prior_latent / (prior_latent.norm(dim=-1, keepdim=True) + 1e-8)
        # Can be extended to learned variances
        kl = 0.5 * torch.sum((latents_norm - prior_norm) ** 2, dim=-1)
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

        latents = latents / (latents.norm(dim=-1, keepdim=True) + 1e-8)            
        # Reconstruction term
        predicted_obs = self.decode_observation(latents)
        reconstruction_loss = F.mse_loss(predicted_obs, observations)
        
        # KL term against diffusion prior
        prior_latents = self.latent_diffusion.sample_latent_prior(batch_size, device)
        kl_loss = self._compute_latent_kl(latents, prior_latents).mean()
        
        # Diffusion score matching loss
        t = torch.randint(0, self.config.diffusion.num_diffusion_steps,
                         (batch_size,), device=device)
        
        noise = torch.randn_like(latents)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize noise
        noisy_latents, _ = self.latent_diffusion.q_sample(latents, t, noise)
        
        predicted_score = self.latent_score_network(
            noisy_latents,
            t.float(),
            observations
        )
        # Add reward prediction loss if rewards provided
        predicted_rewards = self.reward_predictor(latents)
        rewards_mean =predicted_rewards[:, 0]
        rewards_std = torch.exp(torch.clamp(predicted_rewards[:, 1],min=-10, max=3))
        rewards_distribution = torch.distributions.Normal(rewards_mean, rewards_std)
        reward_loss = -rewards_distribution.log_prob(rewards).mean()
        # True score
        sqrt_one_minus_alpha = extract(
            self.latent_diffusion.sqrt_one_minus_alphas_cumprod,
            t,
            latents.shape
        )
        true_score = -noise / sqrt_one_minus_alpha
        
        score_matching_loss = F.mse_loss(predicted_score, true_score)
        
        # Total ELBO
        elbo = -reconstruction_loss + self.config.kl_weight * kl_loss + \
               self.config.diffusion_weight * score_matching_loss - \
               self.config.reward_weight * reward_loss
               
        info = {
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'score_matching_loss': score_matching_loss.item(),
            'elbo': elbo.item(),
            'reward_loss': reward_loss.item(),
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
      
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract coefficients helper"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    

