"""
Score network implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ScoreNetwork(nn.Module):
    """
    Score network s_θ(z,t,π) = ∇_z log p_t(z|π)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_layers: int = 4
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Main network
        layers = []
        input_dim = state_dim + time_embed_dim + hidden_dim
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer to zero
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        
    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score ∇_z log p_t(z|π)
        
        Args:
            state: State z [batch_size, state_dim]
            time: Timestep t [batch_size]
            action: Action/policy π [batch_size, action_dim]
            
        Returns:
            Score [batch_size, state_dim]
        """
        # Embed time and action
        t_emb = self.time_embed(time)
        a_emb = self.action_embed(action)
        
        # Concatenate inputs
        inputs = torch.cat([state, t_emb, a_emb], dim=-1)
        
        # Compute score
        score = self.network(inputs)
        
        return score


