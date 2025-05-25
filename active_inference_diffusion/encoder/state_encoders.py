"""
State and Multi-View Encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

# Import specific classes from visual_encoders
from .visual_encoders import DrQV2Encoder, RandomShiftAugmentation


class StateEncoder(nn.Module):
    """
    Encoder for state-based observations
    Can optionally project to latent space
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_projection: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.use_projection = use_projection
        
        if use_projection and state_dim != latent_dim:
            # Build MLP encoder
            layers = []
            
            for i in range(num_layers):
                if i == 0:
                    layers.extend([
                        nn.Linear(state_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU()
                    ])
                else:
                    layers.extend([
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.ReLU()
                    ])
                    
            layers.extend([
                nn.Linear(hidden_dim, latent_dim),
                nn.Tanh()
            ])
            
            self.encoder = nn.Sequential(*layers)
        else:
            # Identity mapping
            self.encoder = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation"""
        return self.encoder(x)


class MultiViewEncoder(nn.Module):
    """
    Encoder for multiple camera views
    """
    
    def __init__(
        self,
        camera_configs: Dict[str, Dict],
        feature_dim: int = 128,
        frame_stack: int = 1,
        fusion: str = 'attention'  # 'concat', 'sum', 'attention'
    ):
        super().__init__()
        
        self.camera_configs = camera_configs
        self.fusion = fusion
        self.feature_dim = feature_dim
        
        # Create encoder for each camera
        self.encoders = nn.ModuleDict()
        
        for cam_name, config in camera_configs.items():
            self.encoders[cam_name] = DrQV2Encoder(
                obs_shape=(3, config['height'], config['width']),
                feature_dim=feature_dim,
                frame_stack=frame_stack
            )
            
        # Fusion mechanism
        if fusion == 'concat':
            total_features = len(camera_configs) * feature_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(total_features, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh()
            )
        elif fusion == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(feature_dim)
            
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multiple views and fuse them
        
        Args:
            x: Dictionary of camera observations
            
        Returns:
            Fused feature tensor
        """
        features = []
        
        # Encode each view
        for cam_name, encoder in self.encoders.items():
            if cam_name in x:
                feat = encoder(x[cam_name])
                features.append(feat)
                
        if len(features) == 0:
            raise ValueError("No valid camera observations provided")
            
        # Fuse features
        if self.fusion == 'concat':
            fused = torch.cat(features, dim=-1)
            return self.fusion_layer(fused)
            
        elif self.fusion == 'sum':
            return sum(features) / len(features)
            
        elif self.fusion == 'attention':
            # Stack for attention
            stacked = torch.stack(features, dim=1)  # (B, N_views, D)
            
            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # Average pooling
            pooled = attended.mean(dim=1)
            
            return self.fusion_norm(pooled)


class EncoderFactory:
    """Factory for creating encoders"""
    
    @staticmethod
    def create_encoder(
        encoder_type: str,
        obs_shape: Tuple[int, ...],
        feature_dim: int,
        frame_stack: int = 1,
        **kwargs
    ) -> nn.Module:
        """
        Create encoder based on type
        
        Args:
            encoder_type: Type of encoder ('drqv2', 'state', 'multiview')
            obs_shape: Shape of observations
            feature_dim: Output feature dimension
            frame_stack: Number of frames to stack
            **kwargs: Additional arguments
            
        Returns:
            Encoder module
        """
        if encoder_type == 'drqv2':
            return DrQV2Encoder(
                obs_shape=obs_shape,
                feature_dim=feature_dim,
                frame_stack=frame_stack,
                **kwargs
            )
        elif encoder_type == 'state':
            return StateEncoder(
                state_dim=obs_shape[0],
                latent_dim=feature_dim,
                **kwargs
            )
        elif encoder_type == 'multiview':
            return MultiViewEncoder(
                feature_dim=feature_dim,
                frame_stack=frame_stack,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")