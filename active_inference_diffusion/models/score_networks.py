"""
Score Networks for Diffusion-Generated Latent Spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LatentScoreNetwork(nn.Module):
    """
    Score network s_θ(z_t, t, o) = ∇_z log p_t(z|o)
    
    Key innovation: Learns the score function of the latent distribution
    conditioned on observations, enabling context-aware latent generation
    """
    
    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 128,
        num_layers: int = 6,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.use_attention = use_attention
        
        # Time embedding with learnable frequency
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main score network with residual connections
        self.score_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_dim = latent_dim + time_embed_dim + hidden_dim
            else:
                input_dim = hidden_dim
                
            self.score_blocks.append(
                ScoreResidualBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    use_attention=use_attention and i % 2 == 0
                )
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        # Initialize output to zero for stability
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)
        
    def forward(
        self,
        z_t: torch.Tensor,
        time: torch.Tensor,
        observation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute score ∇_z log p_t(z|o)
        
        Args:
            z_t: Noisy latent state [batch_size, latent_dim]
            time: Diffusion timestep [batch_size]
            observation: Conditioning observation [batch_size, obs_dim]
            
        Returns:
            Score [batch_size, latent_dim]
        """
        # Embed time
        t_emb = self.time_embed(time)
        
        # Encode observation
        if observation is not None:
            obs_emb = self.obs_encoder(observation)
        else:
            # Use learned null embedding
            obs_emb = torch.zeros(z_t.shape[0], self.obs_encoder[-1].out_features, 
                                device=z_t.device)
        
        # Concatenate inputs
        h = torch.cat([z_t, t_emb, obs_emb], dim=-1)
        
        # Process through residual blocks
        for block in self.score_blocks:
            h = block(h, t_emb)
        
        # Output score
        score = self.output_proj(h)
        
        return score

class ScoreResidualBlock(nn.Module):
    """Residual block with optional attention for score network"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_attention: bool = False,
        time_embed_dim: int = 128  # Add this parameter
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # First path
        self.norm1 = nn.LayerNorm(hidden_dim if input_dim == hidden_dim else input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.SiLU()
        
        # Time embedding projection
        # This is the key fix - project time embedding to match hidden dimension
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        
        # Second path
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()
        
        # Optional attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            self.norm_attn = nn.LayerNorm(hidden_dim)
        
        # Residual projection if dimensions don't match
        self.residual_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = self.residual_proj(x)
        
        # First transformation
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act1(h)
        
        # Time modulation with projection
        # Project time embedding to match hidden dimension before adding
        time_proj = self.time_proj(time_emb)
        h = h + time_proj
        
        # Second transformation
        h = self.norm2(h)
        h = self.linear2(h)
        h = self.act2(h)
        
        # Optional attention
        if self.use_attention:
            h_attn = self.norm_attn(h)
            h_attn, _ = self.attention(h_attn, h_attn, h_attn)
            h = h + h_attn
        
        return h + residual

class SinusoidalPositionEmbeddings(nn.Module):
    """Enhanced sinusoidal embeddings with learnable frequencies"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable frequency scaling
        self.freq_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # Apply learnable frequency scaling
        embeddings = embeddings * self.freq_scale
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings