"""
Dynamics model implementations
"""

import torch
import torch.nn as nn


class LatentDynamicsModel(nn.Module):
    """
    Latent dynamics model f(s,a) -> s'
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        residual: bool = True
    ):
        super().__init__()
        
        self.residual = residual
        
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output to small values for residual connection
        if residual:
            nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
            nn.init.zeros_(self.network[-1].bias)
            
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next state
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action [batch_size, action_dim]
            
        Returns:
            Next state [batch_size, state_dim]
        """
        inputs = torch.cat([state, action], dim=-1)
        output = self.network(inputs)
        
        if self.residual:
            return state + output
        else:
            return output
