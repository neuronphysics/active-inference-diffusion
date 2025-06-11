"""
Active Inference with Diffusion-Generated Latent Spaces
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
from ..configs import ActiveInferenceConfig
from ..encoder.visual_encoders import ConvDecoder
from .diffusion import LatentDiffusionProcess
from ..models.score_networks import LatentScoreNetwork
from ..models.policy_networks import DiffusionConditionedPolicy
from ..models.value_networks import ValueNetwork
from ..models.dynamics_models import LatentDynamicsModel
from ..utils.util import SpatialAttentionAggregator

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
        config: 'ActiveInferenceConfig',
        pixel_shape: Optional[Tuple[int, int, int]] = None
    ):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.config = config
        self.pixel_shape = pixel_shape 
        self.is_pixel_observation = config.pixel_observation
        self.epistemic_dropout_rate = 0.2
        self.device = torch.device(config.device)
        if self.is_pixel_observation and pixel_shape is not None:
            self.raw_observation_shape = pixel_shape
        else:
            self.raw_observation_shape = None
            
        # Initialize components
        self._build_models()
        self.to(self.device)
        # Current belief state (diffusion-generated)
        self.current_latent = None
        self.latent_trajectory = []
        
    def _build_models(self):
        """Build core models for diffusion active inference"""
   
        # Latent diffusion process
        self.latent_diffusion = LatentDiffusionProcess(
            self.config.diffusion,
            latent_dim=self.latent_dim
        )
        # Add reward preference components
        self.register_buffer('reward_mean', torch.tensor(0.0).to(self.device))
        self.register_buffer('reward_var', torch.tensor(1.0).to(self.device))
        self.register_buffer('preference_temperature', torch.tensor(self.config.preference_temperature).to(self.device))
    

        # Score network for latent generation
        
        self.latent_score_network = LatentScoreNetwork(
            latent_dim=self.latent_dim,
            observation_dim=self.latent_dim,
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
        if not self.is_pixel_observation:
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
            observation_shape = self.observation_dim # For non-pixel observations
        else:
            self.observation_decoder = ConvDecoder(
                latent_dim=self.latent_dim,
                output_dim=np.prod(self.pixel_shape),  # Total pixel count
                img_channels=self.pixel_shape[0],
                hidden_dim=self.config.hidden_dim,
                spatial_size=21,
                )
            # Also add a feature decoder for reconstructing encoded features
            self.feature_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.epistemic_dropout_rate),
                nn.Linear(self.config.hidden_dim, self.latent_dim),  # Decode to feature space
                nn.Tanh()
                )
            observation_shape = self.pixel_shape  # For pixel observations

        # Epistemic estimator for latent uncertainty
        self.epistemic_estimator = FunctionSpaceEpistemicEstimator(
            decoder=self.observation_decoder,
            latent_dim=self.latent_dim,
            observation_shape=observation_shape,
            hidden_dim=self.config.hidden_dim,
            is_pixel_observation=self.is_pixel_observation,
            device=self.device
            )

        # Initialize a reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 2)
        )

    def to(self, device):
        """Override to ensure ALL components move to device"""
        # Convert device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        
        # Call parent to() method
        super().to(device)
        
        # Update our device attribute
        self.device = device
        
        # Explicitly move all components
        self.latent_diffusion = self.latent_diffusion.to(device)
        self.latent_score_network = self.latent_score_network.to(device)
        self.policy_network = self.policy_network.to(device)
        self.value_network = self.value_network.to(device)
        self.latent_dynamics = self.latent_dynamics.to(device)
        self.reward_predictor = self.reward_predictor.to(device)
        
        # Handle observation decoder based on type
        if isinstance(self.observation_decoder, nn.ModuleList):
            # For state observations - move each module in the list
            for i in range(len(self.observation_decoder)):
                self.observation_decoder[i] = self.observation_decoder[i].to(device)
        else:
            # For pixel observations
            self.observation_decoder = self.observation_decoder.to(device)
            
        # Move feature decoder if it exists
        if hasattr(self, 'feature_decoder'):
            self.feature_decoder = self.feature_decoder.to(device)
            
        # Move epistemic estimator with explicit device update
        self.epistemic_estimator = self.epistemic_estimator.to(device)
        self.epistemic_estimator.device = device  # Update its device attribute
        
        # Move buffers
        self.reward_mean = self.reward_mean.to(device)
        self.reward_var = self.reward_var.to(device)
        self.preference_temperature = self.preference_temperature.to(device)
        
        return self
        
    def decode_observation(self, latent: torch.Tensor, decode_to_pixels: bool = True) -> torch.Tensor:
        """
        decoding that can decode to either pixels or features
        
        Args:
            latent: Latent representation
            decode_to_pixels: If True and using pixel observations, decode to raw pixels.
                            If False, decode to encoded feature space.
        """
        latent = latent.to(self.device)
        if self.is_pixel_observation:
            if decode_to_pixels:
                # Decode to pixel space
                return self.observation_decoder(latent)
            else:
                # Decode to feature space (for reconstruction loss)
                self.feature_decoder = self.feature_decoder.to(self.device)
                return self.feature_decoder(latent)
        else:
            # For non-pixel observations, use fully connected decoder
            h = latent
            h1 = self.observation_decoder[0](h)
            h2 = self.observation_decoder[1](h1)
            h2 = h2 + h1  # Skip connection
            h3 = self.observation_decoder[2](h2)
            return self.observation_decoder[3](h3)
        

    def predict_reward_from_latent(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use existing reward predictor to get reward distribution from latent
        """
        latent = latent.to(self.device)
        reward_params = self.reward_predictor(latent)
        reward_mean = reward_params[:, 0]
        reward_std = torch.exp(torch.clamp(reward_params[:, 1], min=-5, max=2))
        return reward_mean, reward_std
    

    def update_belief_via_diffusion(
        self,
        observation: torch.Tensor,
        raw_observation: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Update belief using reverse diffusion process
        This is the core innovation - beliefs as diffusion-generated latents
        """
        observation = observation.to(self.device)
        # Handle different input shapes
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = observation.shape[0]
            
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
        if batch_size == 1:
            latent_mean = self.current_latent.squeeze(0)
            latent_std = torch.zeros_like(latent_mean)  # Single sample, no std
        else:
            latent_mean = self.current_latent.mean(dim=0)
            latent_std = self.current_latent.std(dim=0)
        
        # Decode to observation space for validation
         # Compute reconstruction error appropriately
        if self.is_pixel_observation:
            # For pixel observations, decode to feature space and compare with encoded features
            predicted_features = self.decode_observation(self.current_latent, decode_to_pixels=False)
            reconstruction_error = F.mse_loss(predicted_features, observation)
        else:
            # For state observations, decode to state space
            predicted_obs = self.decode_observation(self.current_latent)
            reconstruction_error = F.mse_loss(predicted_obs, observation)
         
        return {
            'latent': self.current_latent,
            'latent_mean': latent_mean,
            'latent_std': latent_std,
            'trajectory_length': len(trajectory),
            'reconstruction_error': reconstruction_error,
            'observation': observation,
            'raw_observation': raw_observation,
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
        latent = latent.to(self.device)
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

    
                epistemic, epistemic_metrics= self.compute_epistemic_value(
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
            'horizon': horizon,
            **epistemic_metrics
        }
        
        return total_efe, info
    
    def compute_epistemic_value(
        self,   
        next_latent_mean: torch.Tensor,
        next_latent_logvar: torch.Tensor,
        num_samples: int = 5
    ) -> torch.Tensor:
        # Compute epistemic value: H(o|s,π) - H(o|s,θ,π)
        # Epistemic value (ambiguity - observation uncertainty)
        #- H[p(o|s,π)] is entropy marginalizing over model parameters (using dropout)
        #- H[p(o|s,θ,π)] is entropy for a fixed set of parameters
        next_latent_mean = next_latent_mean.to(self.device)
        next_latent_logvar = next_latent_logvar.to(self.device)
        with torch.no_grad():
            epistemic_value, metrics = self.epistemic_estimator(
                next_latent_mean, 
                next_latent_logvar, 
                num_samples
            )
        
        
        return epistemic_value, metrics

    def train_epistemic_estimator(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        next_latents: torch.Tensor
    ) -> float:
        """Train MINE estimator separately"""
        latents = latents.to(self.device)
        actions = actions.to(self.device)
        # Predict next latent distribution
        next_mean, next_logvar = self.predict_next_latent(latents, actions)

        # Compute MINE loss (negative MI for minimization)
        mi_estimate, metrics = self.epistemic_estimator(next_mean, next_logvar)
        loss = -mi_estimate.mean()
    
        # Optimize
        self.epistemic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.epistemic_estimator.parameters(),
            self.config.gradient_clip
        )
        self.epistemic_optimizer.step()

        return mi_estimate.mean().item(), metrics

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
        latent = latent.to(self.device)
        action = action.to(self.device)
        delta = self.latent_dynamics(latent, action)
        next_mean= latent + delta  # Residual connection
        next_logvar = torch.full_like(next_mean, np.log(0.1)).to(self.device)  
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
        deterministic: bool = False,
        raw_observation: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action using diffusion-generated latent active inference
        """
        observation = observation.to(self.device)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Update belief via diffusion
        belief_info = self.update_belief_via_diffusion(observation, raw_observation)
        
        # Current latent belief
        latent = belief_info['latent']
        # Ensure latent has proper shape for policy
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
       
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
        action =action.cpu()
        # Ensure action has proper shape
        if action.dim() > 2:
            action = action.squeeze()
        elif action.dim() == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        elif action.dim() == 0:
            # Handle scalar actions
            action = action.unsqueeze(0)

        
        # Compile information
        info = {
        **belief_info,
        'expected_free_energy': efe.mean().cpu().item(),
        'action_log_prob': log_prob.mean().cpu().item(),
        'policy_entropy': policy_dist.entropy().sum(dim=-1).mean().cpu().item(),
        **{k: v.cpu().item() if torch.is_tensor(v) else v for k, v in efe_info.items()}
        }
        
        return action, info
        
    def compute_diffusion_elbo(
        self,
        observations: torch.Tensor,
        rewards: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        raw_observations: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Annealing time conditioned ELBO for diffusion-generated latents
        Modified ELBO for diffusion-generated latent active inference
        
        L = E_q(z|o,π)[log p(o|z)] - D_KL[q(z|o,π)||p_θ(z)] + R_diffusion(θ)
        """
        observations = observations.to(self.device)
        rewards = rewards.to(self.device)
        batch_size = observations.shape[0]
        device = self.device
        
        # Generate latents if not provided
        if latents is None:
            # Use current belief generation
            belief_info = self.update_belief_via_diffusion(observations, raw_observations)
            latents = belief_info['latent']

                   
        # Reconstruction term
        if self.is_pixel_observation:
            # For pixel observations, reconstruct features
            predicted_features = self.decode_observation(latents, decode_to_pixels=False)
            reconstruction_loss = F.mse_loss(predicted_features, observations)
        else:
            # For state observations, reconstruct states
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

        
        noise = torch.randn_like(latents, device=device)
    
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
        #shape: (batch_size, latent_dim)
        per_sample_losses = loss_weight.view(-1) * torch.sum(score_diff ** 2, dim=1)
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
        lambda_returns = torch.zeros_like(rewards).to(device)
    
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
        noisy_latents = noisy_latents.detach().requires_grad_(True)    
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
        if loss.dim() > 1:
           # If loss has multiple dimensions, reduce to scalar per sample
           loss = loss.view(loss.shape[0], -1).sum(dim=1)

        # Update weights with EMA
        for i in range(len(indices)):  
            time_bin = indices[i].item()
            sample_loss = loss[i].item()
            current_weight = self.time_importance_weights[time_bin].item()
            new_weight = 0.99 * current_weight + 0.01 * sample_loss
            self.time_importance_weights[time_bin] = new_weight

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

class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + 1e-6) / input.shape[0]
        return grad, None

def ema_loss(x, running_mean, alpha=0.01):
    """Exponential moving average loss for stable MINE training"""
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = alpha * t_exp + (1.0 - alpha) * running_mean.item()
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class FunctionSpaceEpistemicEstimator(nn.Module):
    """
    Computes epistemic value I(o; θ | z) via function-space features
    using neural tangent kernel approximation and MINE estimation
    """
    
    def __init__(
        self, 
        decoder: nn.Module,
        latent_dim: int,
        observation_shape: Union[int,Tuple[int, int, int]],
        hidden_dim: int = 256,
        is_pixel_observation: bool = True,
        device: Union[str, torch.device] = 'cuda'
    ):
        super().__init__()
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.is_pixel= is_pixel_observation
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Jacobian approximation parameters
        self.ntk_samples = 5
        self.perturbation_scale = nn.Parameter(torch.tensor(0.1)).to(self.device)
        
        if self.is_pixel:
            self.pixel_shape = observation_shape
            # Pixel-aware MINE architecture using ConvolutionalStatisticsNetwork pattern
            self.pixel_processor = nn.Sequential(
                nn.Conv2d(self.pixel_shape[0], 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
            )
            # Spatial attention aggregator instead of average pooling
            self.spatial_aggregator = SpatialAttentionAggregator(
                feature_dim=128,
                num_heads=8,
                spatial_dim=21  # After 3 stride-2 convolutions from 84x84
            )
            # Jacobian feature projection
            jacobian_dim = hidden_dim * self.ntk_samples  # 128 channels, 2 spatial dimensions (H, W)
        else:
            self.state_dim = observation_shape
            self.feature_extractor = nn.Sequential(
                    nn.Linear(self.state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
            )
            jacobian_dim = 128 * self.ntk_samples

        self.jacobian_projector = nn.Sequential(
            nn.Linear(jacobian_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Latent feature processor
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # MINE statistics network
        self.mine_network = nn.Sequential(
            nn.Linear(hidden_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
        # EMA for stable MINE training
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.alpha = 0.01
        self.to(self.device)

    def to(self, device):
        """Override to ensure proper device movement"""
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # Ensure decoder is also moved properly
        if isinstance(self.decoder, nn.ModuleList):
            for i in range(len(self.decoder)):
                self.decoder[i] = self.decoder[i].to(device)
        else:
            self.decoder = self.decoder.to(device)
            
        return self

    def compute_jacobian_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        Approximates function-space features via finite differences
        in the neural tangent kernel regime
        """
        z = z.to(self.device)
        batch_size = z.shape[0]
        jacobian_samples = []
        decoder_training = self.decoder.training
        self.decoder.eval()  # Ensure decoder is in eval mode
        # Base decoding
        with torch.no_grad():
             
            f_z = self.decoder(z)  # (B, 3, 84, 84)
            if self.is_pixel and f_z.dim() == 4:  # (B, C, H, W)
                f_z_flat = f_z.view(batch_size, -1)
            else:
                f_z_flat = f_z    
        epsilon = self.perturbation_scale    
        # Compute directional derivatives
        for _ in range(self.ntk_samples):
            # Sample perturbation direction
            delta = F.normalize(torch.randn_like(z).to(self.device), dim=-1) * epsilon

            # Compute finite difference
            with torch.no_grad():
                f_z_perturbed = self.decoder(z + delta)
            
                if self.is_pixel and f_z_perturbed.dim() == 4:
                    f_z_perturbed_flat = f_z_perturbed.view(batch_size, -1)
                else:
                    f_z_perturbed_flat = f_z_perturbed
                    
            # Directional derivative
            diff = (f_z_perturbed_flat - f_z_flat) / epsilon
            
            # Process through pixel encoder
            if self.is_pixel:
                # Process pixel differences
                diff_img = diff.view(batch_size, *self.pixel_shape)
                diff_features = self.pixel_processor(diff_img)
                spatial_features, _ = self.spatial_aggregator(diff_features)
                jacobian_samples.append(spatial_features.view(batch_size, -1))
            else:
                # Process state differences
                diff_features = self.feature_extractor(diff)
                jacobian_samples.append(diff_features)

        if decoder_training:
            self.decoder.train()
        # Average Jacobian features
        jacobian_features = torch.cat(jacobian_samples, dim=1)
        return self.jacobian_projector(jacobian_features)
    
    def forward(
        self, 
        next_latent_mean: torch.Tensor,
        next_latent_logvar: torch.Tensor,
        num_samples: int = 5
    ) -> torch.Tensor:
        """
        Estimates epistemic value I(o; θ | z) using MINE
        """
        batch_size = next_latent_mean.shape[0]
        next_latent_mean = next_latent_mean.to(self.device)
        next_latent_logvar = next_latent_logvar.to(self.device)
        
        # Sample latent states
        z_samples = []
        for _ in range(num_samples):
            z = next_latent_mean + torch.randn_like(next_latent_mean).to(self.device) * torch.exp(0.5 * next_latent_logvar)
            z_samples.append(z)
        
        z_all = torch.cat(z_samples, dim=0)  # (B*num_samples, latent_dim)
        
        # Compute Jacobian features (function-space representation)
        jacobian_features = self.compute_jacobian_features(z_all)
        
        # Process latent features
        latent_features = self.latent_processor(z_all)
        
        # Combine features for MINE
        combined_features = torch.cat([jacobian_features, latent_features], dim=1)
        
        # MINE estimation with proper permutation
        t_joint = self.mine_network(combined_features)
        
        # Create marginal by permuting within batch
        jacobian_marginal_list = []
        for i in range(num_samples):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_features = jacobian_features[start_idx:end_idx]
            
            # Shuffle within this batch
            perm = torch.randperm(batch_size, device=self.device)
            jacobian_marginal_list.append(batch_features[perm])
            
        jacobian_marginal = torch.cat(jacobian_marginal_list, dim=0)
        
        # Marginal features
        combined_marginal = torch.cat([jacobian_marginal, latent_features], dim=1)
        t_marginal = self.mine_network(combined_marginal)
        
        # MINE lower bound with EMA
        t_marginal_logsumexp, self.running_mean = ema_loss(
            t_marginal, self.running_mean, self.alpha
        )
        
        mi_lower_bound = t_joint.mean() - t_marginal_logsumexp
        
        # Average over samples and ensure proper shape
        epistemic_value = mi_lower_bound.expand(batch_size)
        
        # Prepare metrics for logging
        metrics = {
            'epistemic/mi_estimate': mi_lower_bound.item(),
            'epistemic/joint_term': t_joint.mean().item(),
            'epistemic/marginal_term': t_marginal_logsumexp.item(),
            'epistemic/running_mean': self.running_mean.item()
        }
    
        return torch.clamp(epistemic_value, min=0.0), metrics
