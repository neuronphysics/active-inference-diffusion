"""
Value network implementations
"""

import torch
import torch.nn as nn
from active_inference_diffusion.models.score_networks import SinusoidalPositionEmbeddings

class ValueNetwork(nn.Module):
    """
    State value function V(s,t)
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )
        
        # Value network
        layers = []
        input_dim = state_dim + time_embed_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Compute state value
        
        Args:
            state: State tensor [batch_size, state_dim]
            time: Time tensor [batch_size]
            
        Returns:
            Value [batch_size, 1]
        """
        t_emb = self.time_embed(time)
        inputs = torch.cat([state, t_emb], dim=-1)
        return self.network(inputs)

 
