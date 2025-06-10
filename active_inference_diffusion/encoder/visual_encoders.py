"""
Enhanced Visual Encoders for pixel-based observations
High-quality architectures with modern techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class DrQV2Encoder(nn.Module):
    """
    Enhanced DrQ-v2 encoder with modern architectural improvements
    
    Key improvements:
    - Spectral normalization for training stability
    - Spatial attention mechanisms
    - Graduated dropout rates
    - Better activation functions (Mish)
    - Group normalization for better batch-size independence
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        feature_dim: int = 50,
        frame_stack: int = 1,
        num_layers: int = 4,
        num_filters: int = 32,
        use_spectral_norm: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Handle frame stacking in channels
        c, h, w = obs_shape
        self.base_channels = c
        self.frame_stack = frame_stack
        self.input_channels = c * frame_stack
        self.use_attention = use_attention
        
        self.obs_shape = (self.input_channels, h, w)
        self.feature_dim = feature_dim
        
        # Build convolutional layers with progressive channel increase
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Channel progression: 32 -> 64 -> 128 -> 256
        channels = [self.input_channels] + [num_filters * (2 ** min(i, 3)) for i in range(num_layers)]
        
        for i in range(num_layers):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            stride = 2 if i == 0 else 1
            
            # Convolutional layer with optional spectral normalization
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False  # No bias when using normalization
            )
            
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            self.convs.append(conv)
            
            # Group normalization (works better than batch norm for RL)
            self.norms.append(nn.GroupNorm(
                num_groups=min(32, out_channels // 4),
                num_channels=out_channels
            ))
            
            # Progressive dropout rates (deeper layers = more dropout)
            dropout_rate = 0.1 * (i / num_layers)  # 0.0 to 0.075
            self.dropouts.append(nn.Dropout2d(dropout_rate))
        
        # Add spatial attention module after conv layers
        if self.use_attention:
            self.attention = SpatialAttention(channels[-1])
        
        # Calculate flattened dimension
        dummy = torch.zeros(1, *self.obs_shape)
        for i, conv in enumerate(self.convs):
            dummy = conv(dummy)
            dummy = self.norms[i](dummy)
            dummy = F.mish(dummy)  # Using Mish activation
            if i < len(self.convs) - 1:  # No dropout on last layer
                dummy = self.dropouts[i](dummy)
        
        if self.use_attention:
            dummy = self.attention(dummy)
            
        self.conv_out_dim = dummy.view(1, -1).shape[1]
        
        # Enhanced output projection with residual
        self.ln = nn.LayerNorm(self.conv_out_dim)
        
        # Multi-layer output projection for better feature extraction
        self.output_layers = nn.Sequential(
            nn.Linear(self.conv_out_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Proper weight initialization for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU family activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with better feature extraction
        
        The encoding process:
        1. Progressive convolutions with increasing channels
        2. Group normalization for stability
        3. Mish activation for smooth gradients
        4. Progressive dropout for regularization
        5. Spatial attention for focusing on important regions
        6. Multi-layer projection for rich features
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
        elif x.dim() == 3:  # Single image without batch
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected observation shape: {x.shape}")
    
        # Normalize if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        # Progressive feature extraction
        for i, (conv, norm, dropout) in enumerate(zip(self.convs, self.norms, self.dropouts)):
            x = conv(x)
            x = norm(x)
            x = F.mish(x)  # Smooth activation function
            
            # Apply dropout (except on last conv layer)
            if i < len(self.convs) - 1:
                x = dropout(x)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            x = self.attention(x)
            
        # Flatten and project
        x = x.view(x.shape[0], -1)
        x = self.ln(x)
        
        # Multi-layer output projection
        features = self.output_layers(x)
        
        return features


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important image regions
    Uses both average and max pooling for robust attention
    """
    
    def __init__(self, channels: int):
        super().__init__()
        
        # Channel reduction for efficiency
        reduced_channels = max(channels // 8, 16)
        
        self.channel_reduce = nn.Conv2d(channels, reduced_channels, 1)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
        # Learnable temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention"""
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and generate attention map
        pool_concat = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.spatial_conv(pool_concat)
        
        # Apply temperature-controlled sigmoid
        attention_map = torch.sigmoid(attention_map / self.temperature)
        
        # Apply attention with residual connection
        attended = x * attention_map
        return x + attended  # Residual for gradient flow


class ConvDecoder(nn.Module):
    """
    High-quality convolutional decoder with progressive upsampling
    
    Key improvements:
    - Sub-pixel convolution for better upsampling
    - Progressive feature refinement
    - Skip connections within decoder
    - Instance normalization for better style consistency
    - Careful dropout placement
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,  # Not used but kept for compatibility
        img_channels: int = 3,
        hidden_dim: int = 256,
        spatial_size: int = 21,  # For 84x84 output
        use_spectral_norm: bool = True,
        device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        self.img_channels = img_channels
        self.device = device
        
        # Initial projection with careful initialization
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 4, hidden_dim * spatial_size * spatial_size),
            nn.LayerNorm(hidden_dim * spatial_size * spatial_size),
            nn.Mish()
        )
        
        # Progressive upsampling decoder
        self.decoder_blocks = nn.ModuleList()
        
        # Block 1: Refine features at 21x21
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                upsample=False,
                use_spectral_norm=use_spectral_norm
            )
        )
        
        # Block 2: Upsample 21x21 -> 42x42
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 2,
                upsample=True,
                use_spectral_norm=use_spectral_norm
            )
        )
        
        # Block 3: Refine at 42x42
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 4,
                upsample=True,
                use_spectral_norm=use_spectral_norm
            )
        )
        
        # Block 4: Upsample 42x42 -> 84x84
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 8,
                upsample=True,
                use_spectral_norm=use_spectral_norm
            )
        )
        

        
        # Output projection with multiple conv layers for refinement
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim // 8, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        self.to(self.device)
        
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Careful weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        High-quality decoding process
        
        The decoder progressively builds up the image:
        1. Project latent to spatial representation
        2. Refine features at low resolution
        3. Progressively upsample and refine
        4. Final multi-layer output projection
        """
        latent = latent.to(self.device)
        batch_size = latent.shape[0]
        
        # Project latent to spatial representation
        h = self.latent_proj(latent)
        h = h.view(batch_size, -1, self.spatial_size, self.spatial_size)
        
        # Progressive decoding with feature refinement
        for block in self.decoder_blocks:
            h = block(h)
        
        # Final output projection
        output = self.output_proj(h)  # (B, 3, 84, 84)
        
        return output


class DecoderBlock(nn.Module):
    """
    High-quality decoder block with multiple architectural improvements
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.upsample = upsample
        
        # Main path with careful design
        layers = []
        
        if upsample:
            # Sub-pixel convolution for better upsampling
            # First increase channels, then pixel shuffle
            layers.append(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
            )
            if use_spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            
            layers.extend([
                nn.PixelShuffle(2),  # Upsample by 2x
                nn.InstanceNorm2d(out_channels),
                nn.Mish()
            ])
        else:
            # Regular convolution for feature refinement
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            
            layers.extend([
                conv,
                nn.InstanceNorm2d(out_channels),
                nn.Mish()
            ])
        
        # Add second conv for more processing
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_spectral_norm:
            conv2 = nn.utils.spectral_norm(conv2)
            
        layers.extend([
            nn.Dropout2d(0.1),
            conv2,
            nn.InstanceNorm2d(out_channels)
        ])
        
        self.main_path = nn.Sequential(*layers)
        
        # Residual path for gradient flow
        if in_channels != out_channels or upsample:
            residual_layers = []
            
            if upsample:
                # Use sub-pixel conv for residual too
                residual_conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
                if use_spectral_norm:
                    residual_conv = nn.utils.spectral_norm(residual_conv)
                    
                residual_layers.extend([
                    residual_conv,
                    nn.PixelShuffle(2)
                ])
            else:
                residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if use_spectral_norm:
                    residual_conv = nn.utils.spectral_norm(residual_conv)
                residual_layers.append(residual_conv)
            
            residual_layers.append(nn.InstanceNorm2d(out_channels))
            self.residual_path = nn.Sequential(*residual_layers)
        else:
            self.residual_path = nn.Identity()
        
        self.activation = nn.Mish()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        main = self.main_path(x)
        residual = self.residual_path(x)
        return self.activation(main + residual)


class RandomShiftAugmentation(nn.Module):
    """
    Enhanced random shift augmentation with smooth boundaries
    """
    
    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = pad
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random shift with reflection padding"""
        if not self.training:
            return x
            
        n, c, h, w = x.shape
        
        # Use reflection padding for more realistic boundaries
        x = F.pad(x, (self.pad,) * 4, mode='reflect')
        
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