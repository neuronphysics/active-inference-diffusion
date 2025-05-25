"""
Free Energy computation module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class FreeEnergyComputation(nn.Module):
    """
    Computes variational free energy for active inference
    
    F = E_q[log q(z) - log p(z,o)]
      = D_KL[q(z)||p(z)] - E_q[log p(o|z)]
      = Complexity - Accuracy
    """
    
    def __init__(self, precision_init: float = 1.0):
        super().__init__()
        
        # Learnable precision (inverse variance of sensory noise)
        self.log_precision = nn.Parameter(torch.log(torch.tensor(precision_init)))
        
    @property
    def precision(self) -> torch.Tensor:
        return torch.exp(self.log_precision)
        
    def compute_loss(
        self,
        states: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        score_network: nn.Module,
        current_time: float = 0.0,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute free energy loss
        
        Args:
            states: Latent states q(z)
            observations: Observations o
            actions: Actions taken
            score_network: Network computing ∇log p(z|π)
            current_time: Current time step
            prior_mean: Mean of prior p(z)
            prior_std: Std of prior p(z)
            
        Returns:
            Free energy loss and component dictionary
        """
        batch_size = states.shape[0]
        device = states.device
        
        # Prior
        if prior_mean is None:
            prior_mean = torch.zeros_like(states)
            
        # Complexity: D_KL[q(z)||p(z)]
        # For Gaussian: 0.5 * sum((μ_q - μ_p)²/σ_p² + σ_q²/σ_p² - 1 - log(σ_q²/σ_p²))
        # Simplified assuming unit variance for q
        complexity = 0.5 * torch.sum(
            (states - prior_mean) ** 2 / (prior_std ** 2),
            dim=-1
        ).mean()
        
        # Accuracy: -E_q[log p(o|z)]
        # Gaussian observation model: -0.5 * precision * ||o - z||²
        observation_error = torch.sum((observations - states) ** 2, dim=-1)
        accuracy = -0.5 * self.precision * observation_error.mean()
        
        # Score matching regularization
        t = torch.full((batch_size,), current_time, device=device)
        score = score_network(states, t, actions)
        score_reg = 0.01 * torch.sum(score ** 2, dim=-1).mean()
        
        # Total free energy
        free_energy = complexity - accuracy + score_reg
        
        info = {
            'complexity': complexity,
            'accuracy': -accuracy,  # Make positive for logging
            'observation_error': observation_error.mean(),
            'score_regularization': score_reg,
            'precision': self.precision
        }
        
        return free_energy, info
        
    def update_precision(self, complexity: torch.Tensor, accuracy: torch.Tensor):
        """
        Update precision based on prediction errors
        Higher complexity relative to accuracy -> decrease precision
        """
        with torch.no_grad():
            precision_error = complexity - accuracy
            self.log_precision.data += 0.01 * precision_error.clamp(-1, 1)
            self.log_precision.data = self.log_precision.data.clamp(-3, 3)


