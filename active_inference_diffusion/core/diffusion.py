"""
Diffusion process implementation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


class DiffusionProcess(nn.Module):
    """
    Implements diffusion process for generative modeling
    Supports various noise schedules and parameterizations
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Setup noise schedule
        self.setup_schedule()
        
    def setup_schedule(self):
        """Initialize noise schedule β_t"""
        steps = self.config.num_diffusion_steps
        
        if self.config.beta_schedule == "linear":
            betas = torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                steps
            )
        elif self.config.beta_schedule == "cosine":
            # Cosine schedule from Nichol & Dhariwal 2021
            s = 0.008
            x = torch.linspace(0, steps, steps + 1)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=0.0001, max=0.999)
        elif self.config.beta_schedule == "sigmoid":
            betas = torch.sigmoid(torch.linspace(-6, 6, steps))
            betas = betas * (self.config.beta_end - self.config.beta_start) + self.config.beta_start
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
            
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def predict_start_from_score(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from score function
        x_0 = (x_t + (1 - ᾱ_t) * score) / √(ᾱ_t)
        """
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t + sqrt_recipm1_alphas_cumprod_t * score
        
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of p(x_{t-1} | x_t) using score
        """
        # Predict x_0
        pred_x_start = self.predict_start_from_score(x_t, t, score)
        
        # Clip predictions
        pred_x_start = torch.clamp(pred_x_start, min=-1, max=1)
        
        # Compute posterior mean
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * pred_x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
        
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) using score
        """
        mean, _, log_variance = self.p_mean_variance(x_t, t, score)
        
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract coefficients at timestep t and reshape to broadcast"""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

