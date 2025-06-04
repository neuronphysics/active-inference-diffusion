"""
Diffusion Process for Latent Space Generation
Implements the theoretical framework for diffusion-generated belief representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import numpy as np
import math


class LatentDiffusionProcess(nn.Module):
    """
    Implements diffusion process specifically for latent space generation
    in active inference framework.
    
    Key innovations:
    - Generates latent belief representations from noise
    - Conditions on observations for context-aware generation
    - Integrates with policy conditioning mechanisms
    """
    
    def __init__(self, config, latent_dim: int = 64):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        
        # Initialize noise schedule
        self.setup_schedule()
        
        # Learnable initial distribution parameters
        self.register_parameter('latent_prior_mean', 
                              nn.Parameter(torch.zeros(self.latent_dim)))
        self.register_parameter('latent_prior_log_std', 
                              nn.Parameter(torch.zeros(self.latent_dim)))
        
        if getattr(config, 'use_positional_embedding', False):
            self.pos_embed = nn.Parameter(torch.zeros(1, self.latent_dim))
            nn.init.normal_(self.pos_embed, std=0.02)        

        # Add continuous time support
        self.continuous_time = True
        self.time_min = 1e-5
        self.time_max = 1.0
        
        # Add learnable log-SNR interpolation
        self.log_snr_min = nn.Parameter(torch.tensor(-10.0))
        self.log_snr_max = nn.Parameter(torch.tensor(10.0))
        
        # Loss weight annealing
        self.register_buffer('loss_weight_cache', torch.zeros(1000))
        self.loss_weight_computed = False
        
    def compute_log_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log signal-to-noise ratio for continuous time"""
        # Interpolate log-SNR based on continuous time
        log_snr = self.log_snr_min + (self.log_snr_max - self.log_snr_min) * (1 - t)
        return log_snr
    
    def continuous_q_sample(
        self,
        z_start: torch.Tensor,
        t: torch.Tensor,  # Now continuous in [0, 1]
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced q_sample for continuous time"""
        if noise is None:
            noise = torch.randn_like(z_start)
            
        # Compute continuous time parameters
        log_snr = self.compute_log_snr(t)
        alpha = torch.sigmoid(log_snr)
        sigma = torch.sigmoid(-log_snr)
        
        # Reshape for broadcasting
        alpha = alpha.view(-1, 1)
        sigma = sigma.view(-1, 1)
        
        # Sample
        z_noisy = torch.sqrt(alpha) * z_start + torch.sqrt(sigma) * noise
        
        # Return additional info for loss computation
        info = {
            'log_snr': log_snr,
            'alpha': alpha,
            'sigma': sigma
        }
        
        return z_noisy, noise, info
    
    def compute_loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Compute annealed loss weight based on time"""
        log_snr = self.compute_log_snr(t)
        
        # Annealing weight: emphasize middle timesteps
        # This helps prevent explosion at t≈0 or t≈1
        weight = torch.exp(-0.5 * (log_snr ** 2) / 4.0)
        
        # Additional stability weight
        time_weight = torch.sin(t * np.pi) + 0.1  # Never zero
        
        return weight * time_weight

    def setup_schedule(self):
        """Enhanced schedule for latent diffusion"""
        steps = self.config.num_diffusion_steps
        
        if self.config.beta_schedule == "cosine":
            # Cosine schedule optimized for latent spaces
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=1e-4, max=0.999)
        elif self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                steps
            )
        else:
            raise ValueError(f"Unknown schedule: {self.config.beta_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register all necessary buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - alphas_cumprod))
        
        # Posterior parameters
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
    def sample_latent_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from learned latent prior p_θ(z)"""
        mean = self.latent_prior_mean.unsqueeze(0).expand(batch_size, -1)
        std = torch.exp(self.latent_prior_log_std).unsqueeze(0).expand(batch_size, -1)
        
        eps = torch.randn_like(mean)
        return mean + std * eps
        
    def q_sample(
        self,
        z_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion for latent representations
        Returns both noisy latent and the noise used
        """
        if noise is None:
            noise = torch.randn_like(z_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, z_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, z_start.shape
        )
        
        z_noisy = sqrt_alphas_cumprod_t * z_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return z_noisy, noise
        
    def generate_latent_trajectory(
        self,
        score_network: nn.Module,
        batch_size: int,
        observation: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> List[torch.Tensor]:
        """
        Generate latent trajectory via reverse diffusion
        This is the core innovation - generating belief representations
        """
        device = next(score_network.parameters()).device
        # Ensure observation is on correct device
        if observation is not None:
            observation = observation.to(device)
    
        z = torch.randn(batch_size, self.latent_dim, device=device)  # Fix shape tuple
        trajectory = [z]
        
        # Reverse diffusion process
        for t in reversed(range(self.config.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict score conditioned on observation
            score = score_network(z, t_batch.float(), observation)
            
            # Update latent
            z = self.p_sample(z, t_batch, score, deterministic=deterministic)
            trajectory.append(z)
            
        return trajectory
        
    def p_sample(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Reverse diffusion step for latent generation
        Implements the score-based update rule
        """
        # Extract parameters
        beta_t = extract(self.betas, t, z_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, z_t.shape
        )
        sqrt_recip_alphas_t = extract(1.0 / torch.sqrt(self.alphas), t, z_t.shape)
        
        # Predict z_0
        predicted_z_start = (z_t + sqrt_one_minus_alphas_cumprod_t * score) * sqrt_recip_alphas_t
        
        # Compute posterior mean
        posterior_mean = self._posterior_mean(predicted_z_start, z_t, t)
        
        if deterministic or t[0] == 0:
            return posterior_mean
        else:
            posterior_variance = extract(self.posterior_variance, t, z_t.shape)
            noise = torch.randn_like(z_t)
            return posterior_mean + torch.sqrt(posterior_variance) * noise
            
    def _posterior_mean(
        self,
        z_start: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute posterior mean for reverse process"""
        posterior_mean_coef1 = extract(
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            t, z_start.shape
        )
        posterior_mean_coef2 = extract(
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod),
            t, z_t.shape
        )
        
        return posterior_mean_coef1 * z_start + posterior_mean_coef2 * z_t


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract coefficients at timestep t"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))