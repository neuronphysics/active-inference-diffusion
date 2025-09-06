"""
Enhanced Visual Encoders for pixel-based observations
High-quality architectures with modern techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np
import math
from typing import Tuple, Optional, Literal
import torch.utils.checkpoint as cp


# ----------------------------- ATTENTION BLOCKS ------------------------------

class GlobalContextAttention2D(nn.Module):
    """
    Cross-attention over image features:
      - Queries:   full-resolution feature map
      - Keys/Vals: pooled (downsampled) feature map
    This provides global context with O(HW * (H'W')) complexity (H'W' << HW).

    Args:
        dim:     channel dim of input features
        heads:   number of attention heads
        dim_head: per-head dimension (total dim = heads * dim_head) after proj
        pool_hw: target pooled spatial size for K/V (int or (h, w))
        dropout: dropout on attention probs and MLP
    """
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        pool_hw: int | Tuple[int, int] = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.q_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.out_proj = nn.Conv2d(inner_dim, dim, 1, bias=False)

        if isinstance(pool_hw, int):
            self.pool_hw = (pool_hw, pool_hw)
        else:
            self.pool_hw = tuple(pool_hw)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5

        # simple LayerNorm over channels via GroupNorm(1, C)
        self.norm_q = nn.GroupNorm(1, dim)
        self.norm_kv = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        b, c, h, w = x.shape

        # pre-norm on channels
        q_in = self.norm_q(x)
        kv_in = self.norm_kv(x)

        # pooled K/V spatial grid
        ph, pw = self.pool_hw
        kv = F.adaptive_avg_pool2d(kv_in, (ph, pw))  # [B, C, ph, pw]

        # projections
        q = self.q_proj(q_in)        # [B, heads*dim_head, H, W]
        k = self.k_proj(kv)          # [B, heads*dim_head, ph, pw]
        v = self.v_proj(kv)          # [B, heads*dim_head, ph, pw]

        # reshape to [B, heads, tokens, dim_head]
        def to_heads(t, H, W):
            t = t.view(b, self.heads, self.dim_head, H * W)  # [B,H,dh,HW]
            return t.permute(0, 1, 3, 2).contiguous()        # [B,H,HW,dh]

        q = to_heads(q, h, w)           # [B, heads, HW, dh]
        k = to_heads(k, ph, pw)         # [B, heads, ph*pw, dh]
        v = to_heads(v, ph, pw)         # [B, heads, ph*pw, dh]

        # attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B,H,HW,ph*pw]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                 # [B,H,HW,dh]
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, h, w)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return x + out  # residual


class MHSA2D(nn.Module):
    """
    Full multi-head self-attention over all spatial tokens.
    Heavier than GlobalContextAttention2D: O((HW)^2).

    Args:
        dim: channels
        heads: number of heads
        dim_head: per-head dim
        dropout: dropout prob
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.out_proj = nn.Conv2d(inner_dim, dim, 1, bias=False)
        self.scale = dim_head ** -0.5
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_n = self.norm(x)
        qkv = self.qkv(x_n)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each [B, inner, H, W]

        def to_heads(t):
            t = t.view(b, self.heads, self.dim_head, h * w)
            return t.permute(0, 1, 3, 2).contiguous()  # [B,H,HW,dh]

        q, k, v = map(to_heads, (q, k, v))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B,H,HW,HW]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                # [B,H,HW,dh]
        out = out.permute(0, 1, 3, 2).contiguous().view(b, self.heads * self.dim_head, h, w)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return x + out


# ------------------------------- ENCODER -------------------------------------

class DrQV2Encoder(nn.Module):
    """
    DrQ-v2 style visual encoder with:
      • Conv trunk (GroupNorm + Mish)
      • Real attention block (choose 'global' or 'mhsa')
      • Projection MLP to feature_dim
      • Optional gradient checkpointing

    Args:
        obs_shape: (C, H, W) base observation shape (pre frame-stack)
        feature_dim: output feature size
        frame_stack: frames to stack on channel dim
        num_layers: #conv layers in trunk
        num_filters: base channels in trunk
        use_spectral_norm: if True, apply SN to convs
        attention: 'none' | 'global' | 'mhsa'
        attn_heads, attn_dim_head: attention config
        attn_pool_hw: pooled grid for global attention (int or (h, w))
        dropout: trunk dropout rate schedule scalar (scaled across layers)
        checkpoint_trunk/attention/head: enable checkpointing for those parts
    """
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        feature_dim: int = 50,
        frame_stack: int = 1,
        num_layers: int = 4,
        num_filters: int = 32,
        use_spectral_norm: bool = False,
        attention: Literal["none", "global", "mhsa"] = "global",
        attn_heads: int = 4,
        attn_dim_head: int = 32,
        attn_pool_hw: int | Tuple[int, int] = 7,
        dropout: float = 0.1,
        checkpoint_trunk: bool = False,
        checkpoint_attention: bool = False,
        checkpoint_head: bool = False,
    ):
        super().__init__()

        c, h, w = obs_shape
        self.base_channels = c
        self.frame_stack = frame_stack
        self.in_channels = c * frame_stack

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.use_spectral_norm = use_spectral_norm
        self.attention_kind = attention
        self.checkpoint_trunk = checkpoint_trunk
        self.checkpoint_attention = checkpoint_attention
        self.checkpoint_head = checkpoint_head

        # ------------- Conv trunk -------------
        chans = [self.in_channels] + [num_filters * (2 ** min(i, 3)) for i in range(num_layers)]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            conv = nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=2 if i == 0 else 1, padding=1, bias=False)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            self.convs.append(conv)

            self.norms.append(nn.GroupNorm(num_groups=min(32, chans[i + 1] // 4), num_channels=chans[i + 1]))
            # progressively increase dropout (none on last conv)
            p = dropout * (i / max(1, num_layers - 1))
            self.dropouts.append(nn.Dropout2d(p if i < num_layers - 1 else 0.0))

        # ------------- Real attention -------------
        if attention == "global":
            self.attn = GlobalContextAttention2D(chans[-1], heads=attn_heads, dim_head=attn_dim_head, pool_hw=attn_pool_hw)
        elif attention == "mhsa":
            self.attn = MHSA2D(chans[-1], heads=attn_heads, dim_head=attn_dim_head)
        else:
            self.attn = nn.Identity()

        # Probe output spatial size + flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, h, w)
            x = dummy
            for i in range(num_layers):
                x = self._trunk_block(x, i)
            x = self.attn(x)
            conv_out_dim = x.view(1, -1).shape[1]

        self.pre_flat_norm = nn.LayerNorm(conv_out_dim)

        # ------------- Projection head -------------
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, feature_dim * 2, bias=True),
            nn.LayerNorm(feature_dim * 2),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim, bias=True),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self._init_weights()

    # ---- init helpers ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- blocks with optional checkpoint ----
    def _maybe_cp(self, fn, x: torch.Tensor, enabled: bool):
        if enabled and isinstance(x, torch.Tensor) and x.requires_grad:
            return cp.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def _trunk_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        conv, norm, drop = self.convs[i], self.norms[i], self.dropouts[i]
        def block(t):
            t = conv(t)
            t = norm(t)
            t = F.mish(t)
            t = drop(t)
            return t
        return self._maybe_cp(block, x, self.checkpoint_trunk)

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_cp(self.attn, x, self.checkpoint_attention)

    def _head_block(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self._maybe_cp(self.head, x_flat, self.checkpoint_head)

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] or [B,T,C,H,W] with T==frame_stack
        returns: [B, feature_dim]
        """
        # frame stack handling
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            assert t == self.frame_stack, f"Expected {self.frame_stack} frames, got {t}"
            x = x.reshape(b, t * c, h, w)
        elif x.dim() == 4:
            b, c, h, w = x.shape
            if c == self.base_channels and self.frame_stack > 1:
                x = x.repeat(1, self.frame_stack, 1, 1)
            elif c != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected observation shape: {x.shape}")

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # conv trunk
        for i in range(self.num_layers):
            x = self._trunk_block(x, i)

        # real attention
        x = self._attn_block(x)

        # flatten + projection
        x = x.view(x.size(0), -1)
        x = self.pre_flat_norm(x)
        feat = self._head_block(x)
        return feat

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
        frame_stack: int = 1,
        device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.spatial_size = spatial_size
        self.img_channels = img_channels
        self.device = device
        self.hidden_dim = hidden_dim
        self.frame_stack = frame_stack

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
        
        # Block 3: Refine at 42x42 -> 84x84
        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 4,
                upsample=True,
                use_spectral_norm=use_spectral_norm
            )
        )
        
        
        # Output projection with multiple conv layers for refinement
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.Mish(),
            nn.Conv2d(32, img_channels * self.frame_stack, kernel_size=3, padding=1),
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
        h = h.view(batch_size, self.hidden_dim, self.spatial_size, self.spatial_size)

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
            # Replace PixelShuffle with: bilinear upsample → 3x3 conv
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if use_spectral_norm:
                conv_up = nn.utils.spectral_norm(conv_up)
            layers.extend([
                conv_up,
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
                # Match main path: bilinear upsample → 1x1 conv
                residual_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
                residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                if use_spectral_norm:
                    residual_conv = nn.utils.spectral_norm(residual_conv)
                residual_layers.append(residual_conv)

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