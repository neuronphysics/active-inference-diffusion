"""
Belief Dynamics using Fokker-Planck equation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
import math
import warnings

class BeliefDynamics(nn.Module):
    """
    Implements continuous belief dynamics using Fokker-Planck equation
    
    The belief distribution q(z|π) evolves according to:
    ∂q/∂t = ∇·(D∇q) + ∇·(qF_π)
    
    For Gaussian approximation:
    dμ/dt = -∇_z F(μ,t)
    dΣ/dt = -ΣH - HΣ + 2DI
    
    where F is the free energy gradient and H is its Hessian.
    Implements continuous belief dynamics using Fokker-Planck equation
    with enhanced numerical stability and theoretical consistency
    
    Enhanced Features:
    - Matrix exponential-based covariance updates
    - Automatic regularization for ill-conditioned matrices
    - Gradient-based Hessian approximation using automatic differentiation
    - Condition number monitoring and adaptive stabilization
    """
    
    def __init__(self, latent_dim: int, config):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.config = config
        
        # Initialize belief parameters with enhanced precision
        self.register_buffer('mean', torch.zeros(latent_dim, dtype=torch.float64))
        
        if config.use_full_covariance:
            # Use double precision for matrix operations
            self.register_buffer('covariance', torch.eye(latent_dim, dtype=torch.float64))
            self.register_buffer('precision', torch.eye(latent_dim, dtype=torch.float64))
        else:
            # Diagonal covariance for efficiency
            self.register_buffer('variance', torch.ones(latent_dim, dtype=torch.float64))
            self.register_buffer('precision', torch.ones(latent_dim, dtype=torch.float64))
            
        # Enhanced history tracking with numerical diagnostics
        self.history = {
            'means': [],
            'covariances': [],
            'entropies': [],
            'free_energies': [],
            'condition_numbers': [],
            'numerical_warnings': []
        }
        
        # Numerical stability parameters
        self.min_eigenvalue = max(config.min_variance, 1e-8)
        self.max_condition_number = 1e6
        
    def reset(
        self,
        initial_mean: Optional[torch.Tensor] = None,
        initial_cov: Optional[torch.Tensor] = None
    ):
        """Reset belief to initial state with enhanced initialization"""
        if initial_mean is not None:
            self.mean = initial_mean.to(self.mean.device).to(torch.float64)
        else:
            self.mean.zero_()
            
        if self.config.use_full_covariance:
            if initial_cov is not None:
                self.covariance = initial_cov.to(self.covariance.device).to(torch.float64)
                self.covariance = self._ensure_numerical_stability(self.covariance)
                self.precision = self._safe_inverse(self.covariance)
            else:
                self.covariance = torch.eye(self.latent_dim, device=self.mean.device, dtype=torch.float64)
                self.precision = torch.eye(self.latent_dim, device=self.mean.device, dtype=torch.float64)
        else:
            if initial_cov is not None:
                self.variance = torch.diag(initial_cov).to(self.variance.device).to(torch.float64)
                self.variance = torch.clamp(self.variance, min=self.min_eigenvalue)
                self.precision = 1.0 / self.variance
            else:
                self.variance.fill_(1.0)
                self.precision.fill_(1.0)
                
        # Clear history
        self.history = {key: [] for key in self.history.keys()}
        
    def update(
        self,
        observation: torch.Tensor,
        score_function: torch.Tensor,
        action: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update belief using enhanced Fokker-Planck dynamics
        
        Implements numerically stable matrix exponential method for covariance evolution:
        Σ(t+dt) = exp([A_drift + A_diffusion] * dt) @ Σ(t)
        """
        dt = self.config.dt
        D = self.config.diffusion_coefficient
        lr = self.config.learning_rate
        
        # Convert inputs to double precision for stability
        observation = observation.to(torch.float64)
        score_function = score_function.to(torch.float64)
        
        # Ensure proper dimensions
        if observation.dim() > 1:
            observation = observation.squeeze(0)
        if score_function.dim() > 1:
            score_function = score_function.squeeze(0)
            
        # Compute free energy gradient with automatic differentiation
        F_gradient = self._compute_free_energy_gradient_autodiff(
            self.mean, observation, score_function, observation_model
        )
        
        # Enhanced mean update with adaptive step size
        mean_drift = -lr * F_gradient
        mean_noise = math.sqrt(2 * D * dt) * torch.randn_like(self.mean) * self.config.noise_scale
        
        # Adaptive step size based on gradient magnitude
        grad_norm = F_gradient.norm()
        adaptive_dt = dt / (1 + 0.1 * grad_norm)  # Reduce step size for large gradients
        
        self.mean = self.mean + mean_drift * adaptive_dt + mean_noise
        
        # Enhanced covariance update
        if self.config.use_full_covariance:
            # Compute Hessian using automatic differentiation
            H = self._compute_hessian_autodiff(
                self.mean, observation, score_function, observation_model
            )
            
            # Matrix exponential-based update for numerical stability
            self.covariance = self._matrix_exponential_update(H, dt, D)
            
            # Ensure numerical stability
            self.covariance = self._ensure_numerical_stability(self.covariance)
            self.precision = self._safe_inverse(self.covariance)
            
        else:
            # Enhanced diagonal update
            H_diag = self._compute_diagonal_hessian_autodiff(
                self.mean, observation, score_function, observation_model
            )
            
            # Exponential update for diagonal case
            var_update_factor = torch.exp((-2 * H_diag + 2 * D) * dt)
            self.variance = self.variance * var_update_factor
            
            # Clamp to ensure stability
            self.variance = torch.clamp(
                self.variance, self.min_eigenvalue, self.config.max_variance
            )
            self.precision = 1.0 / self.variance
            
        # Record enhanced state with diagnostics
        self._record_state_enhanced(observation)
        
        return self.get_parameters()
    
    def _compute_free_energy_gradient_autodiff(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute free energy gradient using automatic differentiation
        Enhanced numerical stability and theoretical consistency
        """
        z_var = z.clone().detach().requires_grad_(True)
        
        # Observation likelihood term
        if observation_model is not None:
            log_p_o_given_z = observation_model(z_var.unsqueeze(0), observation.unsqueeze(0))
            obs_log_prob = log_p_o_given_z.squeeze()
        else:
            # Enhanced Gaussian observation model with learnable precision
            obs_error = torch.sum((z_var - observation) ** 2)
            obs_log_prob = -0.5 * obs_error / (self.config.noise_scale ** 2)
        
        # Prior term (standard normal)
        prior_log_prob = -0.5 * torch.sum(z_var ** 2)
        
        # Score term (from diffusion model)
        score_term = torch.sum(z_var * score)
        
        # Total log probability
        total_log_prob = obs_log_prob + prior_log_prob + score_term
        
        # Compute gradient
        gradient = torch.autograd.grad(
            total_log_prob, z_var, create_graph=False, retain_graph=False
        )[0]
        
        return gradient.detach()
    
    def _compute_hessian_autodiff(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute Hessian using automatic differentiation
        More accurate than finite differences
        """
        z_var = z.clone().detach().requires_grad_(True)
        
        # Compute gradient first
        gradient = self._compute_free_energy_gradient_autodiff(
            z_var, observation, score, observation_model
        )
        
        # Compute Hessian
        H = torch.zeros(self.latent_dim, self.latent_dim, device=z.device, dtype=torch.float64)
        
        for i in range(self.latent_dim):
            grad_i = torch.autograd.grad(
                gradient[i], z_var, create_graph=False, retain_graph=True
            )[0]
            H[i, :] = grad_i
            
        # Symmetrize for numerical stability
        H = 0.5 * (H + H.T)
        
        return H
    
    def _compute_diagonal_hessian_autodiff(
        self,
        z: torch.Tensor,
        observation: torch.Tensor,
        score: torch.Tensor,
        observation_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Compute only diagonal elements of Hessian for efficiency"""
        z_var = z.clone().detach().requires_grad_(True)
        
        gradient = self._compute_free_energy_gradient_autodiff(
            z_var, observation, score, observation_model
        )
        
        diag = torch.zeros(self.latent_dim, device=z.device, dtype=torch.float64)
        
        for i in range(self.latent_dim):
            grad_ii = torch.autograd.grad(
                gradient[i], z_var, create_graph=False, retain_graph=True
            )[0][i]
            diag[i] = grad_ii
            
        return diag
    
    def _matrix_exponential_update(
        self, 
        H: torch.Tensor, 
        dt: float, 
        D: float
    ) -> torch.Tensor:
        """
        Numerically stable covariance update using matrix exponential
        
        Implements: Σ(t+dt) = exp((-H - H^T + 2DI) * dt) @ Σ(t)
        """
        # Construct drift matrix
        drift_matrix = -H - H.T + 2 * D * torch.eye(
            self.latent_dim, device=H.device, dtype=H.dtype
        )
        
        # Compute matrix exponential
        try:
            exp_drift = torch.matrix_exp(drift_matrix * dt)
            new_covariance = exp_drift @ self.covariance @ exp_drift.T
        except RuntimeError as e:
            warnings.warn(f"Matrix exponential failed: {e}, using first-order approximation")
            # Fallback to first-order approximation
            approx_update = torch.eye(self.latent_dim, device=H.device, dtype=H.dtype) + drift_matrix * dt
            new_covariance = approx_update @ self.covariance @ approx_update.T
        
        return new_covariance
    
    def _ensure_numerical_stability(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Enhanced positive definiteness with condition number control
        """
        try:
            # Eigenvalue decomposition
            eigvals, eigvecs = torch.linalg.eigh(matrix)
            
            # Clamp eigenvalues
            eigvals_clamped = torch.clamp(eigvals, min=self.min_eigenvalue)
            
            # Check condition number
            condition_number = eigvals_clamped.max() / eigvals_clamped.min()
            
            if condition_number > self.max_condition_number:
                # Regularize eigenvalues to improve conditioning
                regularization = eigvals_clamped.mean() * 1e-6
                eigvals_clamped = eigvals_clamped + regularization
                
                self.history['numerical_warnings'].append(
                    f"High condition number {condition_number:.2e}, regularized"
                )
            
            # Reconstruct matrix
            stabilized = eigvecs @ torch.diag(eigvals_clamped) @ eigvecs.T
            
            # Record condition number
            self.history['condition_numbers'].append(condition_number.item())
            
            return stabilized
            
        except RuntimeError as e:
            warnings.warn(f"Eigenvalue decomposition failed: {e}, using diagonal regularization")
            # Fallback: add small diagonal regularization
            return matrix + self.min_eigenvalue * torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            )
    
    def _safe_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
        """Numerically stable matrix inversion"""
        try:
            return torch.linalg.inv(matrix + self.min_eigenvalue * torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype
            ))
        except RuntimeError:
            warnings.warn("Matrix inversion failed, using pseudo-inverse")
            return torch.linalg.pinv(matrix)
    
    def _record_state(self, observation: torch.Tensor):
        """Enhanced state recording with numerical diagnostics"""
        # Convert back to float32 for storage efficiency
        self.history['means'].append(self.mean.detach().cpu().to(torch.float32))
        
        if self.config.use_full_covariance:
            self.history['covariances'].append(self.covariance.detach().cpu().to(torch.float32))
        else:
            self.history['covariances'].append(torch.diag(self.variance).detach().cpu().to(torch.float32))
            
        self.history['entropies'].append(self.entropy().detach().cpu().to(torch.float32))
        
        # Enhanced free energy computation
        obs_error = torch.sum((self.mean - observation) ** 2)
        free_energy = -self.entropy() - 0.5 * obs_error / (self.config.noise_scale ** 2)
        self.history['free_energies'].append(free_energy.detach().cpu().to(torch.float32))
    
    def get_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current belief parameters in float32 for downstream compatibility"""
        if self.config.use_full_covariance:
            return self.mean.to(torch.float32), self.covariance.to(torch.float32)
        else:
            return self.mean.to(torch.float32), torch.diag(self.variance).to(torch.float32)
    
    def entropy(self) -> torch.Tensor:
        """Compute entropy with enhanced numerical stability"""
        k = self.latent_dim
        
        if self.config.use_full_covariance:
            # Enhanced log determinant computation
            try:
                log_det = torch.logdet(self.covariance)
                if torch.isnan(log_det) or torch.isinf(log_det):
                    # Fallback to eigenvalue-based computation
                    eigvals = torch.linalg.eigvals(self.covariance).real
                    log_det = torch.sum(torch.log(torch.clamp(eigvals, min=self.min_eigenvalue)))
            except RuntimeError:
                log_det = k * math.log(self.min_eigenvalue)  # Conservative fallback
                
            entropy = 0.5 * (k * math.log(2 * math.pi * math.e) + log_det)
        else:
            # Diagonal case
            log_vars = torch.log(torch.clamp(self.variance, min=self.min_eigenvalue))
            entropy = 0.5 * torch.sum(math.log(2 * math.pi * math.e) + log_vars)
            
        return entropy
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Comprehensive numerical diagnostics"""
        diagnostics = {}
        
        if self.config.use_full_covariance:
            eigvals = torch.linalg.eigvals(self.covariance).real
            diagnostics['min_eigenvalue'] = eigvals.min().item()
            diagnostics['max_eigenvalue'] = eigvals.max().item()
            diagnostics['condition_number'] = (eigvals.max() / eigvals.min()).item()
            diagnostics['determinant'] = torch.det(self.covariance).item()
        else:
            diagnostics['min_variance'] = self.variance.min().item()
            diagnostics['max_variance'] = self.variance.max().item()
            diagnostics['mean_variance'] = self.variance.mean().item()
            
        diagnostics['mean_norm'] = self.mean.norm().item()
        diagnostics['entropy'] = self.entropy().item()
        
        return diagnostics