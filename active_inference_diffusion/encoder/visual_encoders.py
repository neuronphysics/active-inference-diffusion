"""
Visual Encoders for pixel-based observations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class DrQV2Encoder(nn.Module):
    """
    DrQ-v2 encoder - best for pixel-based control
    Handles frame stacking properly
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        feature_dim: int = 50,
        frame_stack: int = 1,
        num_layers: int = 4,
        num_filters: int = 32
    ):
        super().__init__()
        
        # Handle frame stacking in channels
        c, h, w = obs_shape
        self.base_channels = c
        self.frame_stack = frame_stack
        self.input_channels = c * frame_stack  # This fixes the issue!
        
        self.obs_shape = (self.input_channels, h, w)
        self.feature_dim = feature_dim
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = self.input_channels if i == 0 else num_filters
            stride = 2 if i == 0 else 1
            
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    num_filters,
                    kernel_size=3,
                    stride=stride,
                    padding=1
                )
            )
            
        # Calculate flattened dimension
        dummy = torch.zeros(1, *self.obs_shape)
        for conv in self.convs:
            dummy = F.relu(conv(dummy))
        self.conv_out_dim = dummy.view(1, -1).shape[1]
        
        # Output layers
        self.ln = nn.LayerNorm(self.conv_out_dim)
        self.linear = nn.Linear(self.conv_out_dim, feature_dim)
        self.output_ln = nn.LayerNorm(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C*frame_stack, H, W) or (B, C, H, W)
            
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Handle different input formats
        if x.dim() == 5:  # (B, T, C, H, W) - separate frames
            b, t, c, h, w = x.shape
            assert t == self.frame_stack, f"Expected {self.frame_stack} frames, got {t}"
            x = x.reshape(b, t * c, h, w)
        elif x.dim() == 4:  # (B, C, H, W) - already stacked or single frame
            b, c, h, w = x.shape
            if c == self.base_channels and self.frame_stack > 1:
                # Single frame when expecting stack - repeat
                x = x.repeat(1, self.frame_stack, 1, 1)
            elif c != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} channels, got {c}")
                
        # Normalize if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        # Conv layers
        for conv in self.convs:
            x = F.relu(conv(x))
            
        # Flatten and project
        x = x.view(x.shape[0], -1)
        x = self.ln(x)
        x = self.linear(x)
        x = self.output_ln(x)
        x = torch.tanh(x)
        
        return x


class RandomShiftAugmentation(nn.Module):
    """
    Random shift augmentation for data efficiency
    """
    
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random shift"""
        if not self.training:
            return x
            
        n, c, h, w = x.shape
        
        # Pad
        x = F.pad(x, (self.pad,) * 4, mode='replicate')
        
        # Random crop positions
        h_offset = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
        w_offset = torch.randint(0, 2 * self.pad + 1, (n,), device=x.device)
        
        # Crop each image in the batch
        cropped = []
        for i in range(n):
            cropped.append(
                x[i, :, h_offset[i]:h_offset[i]+h, w_offset[i]:w_offset[i]+w]
            )
            
        return torch.stack(cropped)

