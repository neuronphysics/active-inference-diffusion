"""
Belief Dynamics using Fokker-Planck equation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import math


class BeliefDynamics(nn.Module):
    """
    Implements continuous belief dynamics using Fokker-Planck equation
    
    The belief distribution q(z|π) evolves according to:
    ∂q/∂t = ∇·(D∇q) + ∇·(qF_π)
    
    For Gaussian approximation:
    dμ/dt = -∇_z F(μ,t)
    dΣ/dt = -ΣH - HΣ + 2DI
    
    where F is the free energy gradient and H is its Hessian.
    """
    
    def __init__(self, latent_dim: int, config):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.config = config
        
        # Initialize belief parameters
        self.register_buffer('mean', torch.zeros(latent_dim))
        
        if config.use_full_covariance:
            self.register_buffer('covariance', torch.eye(latent_dim))
            self.register_buffer('precision', torch.eye(latent_dim))
        else:
            # Diagonal covariance for efficiency
            self.register_buffer('variance', torch.ones(latent_dim))
            self.register_buffer('precision', torch.ones(latent_dim))
            
        # History tracking
        self.history = {
            'means': [],
            'covariances': [],
            'entropies': [],
            'free_energies': []
        }
        
    def reset(
        self,
        initial_mean: Optional[torch.Tensor] = None,
        initial_cov: Optional[torch.Tensor] = None
    ):
        """Reset belief to initial state"""
        if initial_mean is not None:
            self.mean = initial_mean.to(self.mean.device)
        else:
            self.mean.zero_()
            
        if self.config.use_full_covariance:
            if initial_cov is not None:
                self.covariance = initial_cov.to(self.covariance.device)
                self.precision = torch.linalg.inv(
                    self.covariance + self.config.min_variance * torch.eye(
                        self.latent_dim, device=self.covariance.device
                    )
                )
            else:
                self.covariance = torch.eye(self.latent_dim, device=self.mean.device)
                self.precision = torch.eye(self.latent_dim, device=self.mean.device)
        else:
            if initial_cov is not None:
                self.variance = torch.diag(initial_cov).to(self.variance.device)
                self.precision = 1.0 / (self.variance + self.config.min_variance)
            else:
                self.variance.fill_(1.0)
                self.precision.fill_(1.0)
                
        # Clear history
        self.history = {
            'means': [],
            'covariances': [],
            'entropies': [],
            'free_energies': []
        }
        
    def update(
        self,
        observation: torch.Tensor,
        score_function: torch.Tensor,
        action: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update belief using Fokker-Planck dynamics
        
        Args:
            observation: Observed state (latent encoding)
            score_function: Score ∇log p(z|π) at current belief
            action: Current action
            observation_model: Optional learned observation model
            
        Returns:
            Updated mean and covariance
        """
        dt = self.config.dt
        D = self.config.diffusion_coefficient
        lr = self.config.learning_rate
        
        # Ensure proper dimensions
        if observation.dim() > 1:
            observation = observation.squeeze(0)
        if score_function.dim() > 1:
            score_function = score_function.squeeze(0)
            
        # Compute free energy gradient F = ∇_z F
        F_gradient = self._compute_free_energy_gradient(
            self.mean,
            observation,
            score_function,
            observation_model
        )
        
        # Update mean: dμ/dt = -∇F(μ,t)
        mean_drift = -lr * F_gradient
        mean_noise = math.sqrt(2 * D * dt) * torch.randn_like(self.mean) * self.config.noise_scale
        
        self.mean = self.mean + mean_drift * dt + mean_noise
        
        # Update covariance
        if self.config.use_full_covariance:
            # Approximate Hessian
            H = self._approximate_hessian(
                self.mean, observation, score_function, observation_model
            )
            
            # dΣ/dt = -ΣH - HΣ + 2DI
            cov_drift = -torch.matmul(self.covariance, H) - torch.matmul(H, self.covariance)
            cov_diffusion = 2 * D * torch.eye(self.latent_dim, device=self.mean.device)
            
            self.covariance = self.covariance + (cov_drift + cov_diffusion) * dt
            
            # Ensure positive definiteness
            self.covariance = self._ensure_positive_definite(self.covariance)
            self.precision = torch.linalg.inv(
                self.covariance + self.config.min_variance * torch.eye(
                    self.latent_dim, device=self.covariance.device
                )
            )
        else:
            # Diagonal update (more efficient)
            H_diag = self._approximate_diagonal_hessian(
                self.mean, observation, score_function, observation_model
            )
            
            var_drift = -2 * self.variance * H_diag
            var_diffusion = 2 * D * torch.ones_like(self.variance)
            
            self.variance = self.variance + (var_drift + var_diffusion) * dt
            self.variance = torch.clamp(
                self.variance,
                self.config.min_variance,
                self.config.max_variance
            )
            self.precision = 1.0 / (self.variance + self.config.min_variance)
            
        # Record history
        self._record_state(observation)
        
        return self.get_parameters()
        
    def _compute_free_energy_gradient(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute free energy gradient
        F = ∇log p(o|z) - ∇log q(z|π) + ∇log p(z)
        """
        # Observation likelihood gradient
        if observation_model is not None:
            # Use learned model
            z_grad = z.requires_grad_(True)
            log_p_o_given_z = observation_model(
                z_grad.unsqueeze(0),
                observation.unsqueeze(0)
            )
            obs_gradient = torch.autograd.grad(
                log_p_o_given_z.sum(), z_grad, create_graph=True
            )[0]
        else:
            # Gaussian observation model
            obs_gradient = -(z - observation) / (self.config.noise_scale ** 2)
            
        # Prior gradient (standard normal)
        prior_gradient = -z
        
        # Free energy gradient
        F_gradient = obs_gradient - score + 0.01 * prior_gradient
        
        return F_gradient
        
    def _approximate_hessian(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Approximate Hessian using finite differences"""
        eps = 1e-4
        H = torch.zeros(self.latent_dim, self.latent_dim, device=z.device)
        
        # Base gradient
        base_grad = self._compute_free_energy_gradient(
            z, observation, score, observation_model
        )
        
        # Finite differences
        for i in range(self.latent_dim):
            z_plus = z.clone()
            z_plus[i] += eps
            
            # Perturbed score (approximate)
            score_plus = score + eps * torch.randn_like(score) * 0.1
            
            grad_plus = self._compute_free_energy_gradient(
                z_plus, observation, score_plus, observation_model
            )
            
            H[i, :] = (grad_plus - base_grad) / eps
            
        # Symmetrize
        H = 0.5 * (H + H.T)
        
        return H
        
    def _approximate_diagonal_hessian(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Approximate only diagonal elements of Hessian"""
        eps = 1e-4
        diag = torch.zeros(self.latent_dim, device=z.device)
        
        base_grad = self._compute_free_energy_gradient(
            z, observation, score, observation_model
        )
        
        for i in range(self.latent_dim):
            z_plus = z.clone()
            z_plus[i] += eps
            
            score_plus = score.clone()
            score_plus[i] += eps * 0.1
            
            grad_plus = self._compute_free_energy_gradient(
                z_plus, observation, score_plus, observation_model
            )
            
            diag[i] = (grad_plus[i] - base_grad[i]) / eps
            
        return diag
        
    def _ensure_positive_definite(self, matrix: torch.Tensor) -> torch.Tensor:
        """Ensure matrix is positive definite"""
        # Eigenvalue decomposition
        eigvals, eigvecs = torch.linalg.eigh(matrix)
        
        # Clamp eigenvalues
        eigvals = torch.clamp(eigvals, min=self.config.min_variance)
        
        # Reconstruct
        return eigvecs @ torch.diag(eigvals) @ eigvecs.T
        
    def _record_state(self, observation: torch.Tensor):
        """Record current belief state for visualization"""
        self.history['means'].append(self.mean.detach().cpu())
        
        if self.config.use_full_covariance:
            self.history['covariances'].append(self.covariance.detach().cpu())
        else:
            self.history['covariances'].append(torch.diag(self.variance).detach().cpu())
            
        self.history['entropies'].append(self.entropy().detach().cpu())
        
        # Compute free energy (simplified)
        obs_error = torch.sum((self.mean - observation) ** 2)
        free_energy = -self.entropy() - 0.5 * obs_error / (self.config.noise_scale ** 2)
        self.history['free_energies'].append(free_energy.detach().cpu())
        
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current belief parameters"""
        if self.config.use_full_covariance:
            return self.mean, self.covariance
        else:
            return self.mean, torch.diag(self.variance)
            
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample from current belief distribution"""
        if self.config.use_full_covariance:
            # Multivariate normal
            L = torch.linalg.cholesky(
                self.covariance + self.config.min_variance * torch.eye(
                    self.latent_dim, device=self.covariance.device
                )
            )
            z = torch.randn(n_samples, self.latent_dim, device=self.mean.device)
            samples = self.mean + (z @ L.T)
        else:
            # Independent normal
            std = torch.sqrt(self.variance)
            samples = self.mean + torch.randn(
                n_samples, self.latent_dim, device=self.mean.device
            ) * std
            
        return samples
        
    def entropy(self) -> torch.Tensor:
        """Compute entropy of current belief"""
        k = self.latent_dim
        
        if self.config.use_full_covariance:
            # H = 0.5 * (k * log(2πe) + log|Σ|)
            log_det = torch.logdet(self.covariance)
            entropy = 0.5 * (k * math.log(2 * math.pi * math.e) + log_det)
        else:
            # H = 0.5 * Σ log(2πeσ²)
            entropy = 0.5 * torch.sum(
                torch.log(2 * math.pi * math.e * self.variance)
            )
            
        return entropy
        
    def kl_divergence(
        self,
        other_mean: torch.Tensor,
        other_cov: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence to another Gaussian"""
        k = self.latent_dim
        
        if self.config.use_full_covariance:
            # Full covariance KL
            other_precision = torch.linalg.inv(other_cov)
            mean_diff = other_mean - self.mean
            
            trace_term = torch.trace(other_precision @ self.covariance)
            quad_term = mean_diff @ other_precision @ mean_diff
            log_det_term = torch.logdet(other_cov) - torch.logdet(self.covariance)
            
            kl = 0.5 * (trace_term + quad_term - k + log_det_term)
        else:
            # Diagonal KL
            other_var = torch.diag(other_cov)
            mean_diff = other_mean - self.mean
            
            trace_term = torch.sum(self.variance / other_var)
            quad_term = torch.sum(mean_diff ** 2 / other_var)
            log_det_term = torch.sum(torch.log(other_var / self.variance))
            
            kl = 0.5 * (trace_term + quad_term - k + log_det_term)
            
        return kl