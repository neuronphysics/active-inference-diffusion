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
        
        # DiT configuration
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.use_attention = use_attention
        # Time embedding - FIXED to output hidden_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim*2),  # Changed to output hidden_dim directly
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Observation encoder (unchanged)
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Input projection for latent
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        if use_attention:
            # DiT Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,

                )
                for _ in range(num_layers)
            ])
        
        # Output projection with adaptive layer norm
        self.norm_final = AdaptiveLayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, latent_dim, bias=False)
        )
        
        # Initialize output to zero for stability
        nn.init.zeros_(self.output_proj[-1].weight)
        
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
        batch_size = z_t.shape[0]
        # Encode observation
        if observation is not None:
            obs_emb = self.obs_encoder(observation)
        else:
            # Use learned null embedding
            obs_emb = torch.zeros(batch_size, self.obs_encoder[-1].out_features, 
                                device=z_t.device)
        
        # Combine conditioning (time + observation)
        # This will be used for adaptive normalization in DiT blocks
        conditioning = t_emb + obs_emb  # [B, hidden_dim]
        
        # Project latent to hidden dimension
        h = self.latent_proj(z_t)  # [B, hidden_dim]

        if self.use_attention:
            # Process through DiT blocks
            for block in self.transformer_blocks:
                h = block(h, conditioning)
        
        # Final norm and output
        h = self.norm_final(h, conditioning)
        score = self.output_proj(h)
        
        return score


class DiTBlock(nn.Module):
    """
    Diffusion Transformer block with adaptive layer norm
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Adaptive layer norms (modulated by conditioning)
        self.norm1 = AdaptiveLayerNorm(hidden_dim)
        self.norm2 = AdaptiveLayerNorm(hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )
        
        # Initialize weights for better training stability
        self._init_weights()
        
    def _init_weights(self):
        # Initialize MLP
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.xavier_uniform_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditioning
        
        Args:
            x: Input features [B, hidden_dim]
            conditioning: Time + observation embedding [B, hidden_dim]
        """
        # Self-attention with adaptive norm
        norm_x = self.norm1(x, conditioning)
        # For single token (no sequence), we need to add sequence dimension
        norm_x = norm_x.unsqueeze(1)  # [B, 1, hidden_dim]
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        attn_out = attn_out.squeeze(1)  # [B, hidden_dim]
        x = x + attn_out
        
        # MLP with adaptive norm
        norm_x = self.norm2(x, conditioning)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization for conditioning in DiT
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Projection for adaptive parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )
        
        # Initialize modulation to identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer norm
        
        Args:
            x: Input features [B, hidden_dim]
            conditioning: Conditioning features [B, hidden_dim]
        """
        # Get scale and shift from conditioning
        scale_shift = self.adaLN_modulation(conditioning)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Apply normalization with adaptive scale and shift
        return self.norm(x) * (1 + scale) + shift


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