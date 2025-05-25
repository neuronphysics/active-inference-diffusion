"""
Policy network implementations
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple

class GaussianPolicy(nn.Module):
    """
    Gaussian policy for continuous control
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(state_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            
        self.shared = nn.Sequential(*layers)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> dist.Normal:
        """
        Forward pass returning action distribution
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Normal distribution over actions
        """
        features = self.shared(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return dist.Normal(mean, std)
        
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and log probability
        """
        dist = self.forward(state)
        
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
            
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


