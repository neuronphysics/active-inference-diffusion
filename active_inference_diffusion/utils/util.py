import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # add at top of file or inside function

matplotlib.use("Agg")
class SpatialAttentionAggregator(nn.Module):
    """
    Multi-head attention for spatially-aware epistemic feature aggregation
    Preserves and weights spatial information based on uncertainty relevance
    """
    
    def __init__(self, feature_dim: int = 128, num_heads: int = 8, spatial_dim: int = 21):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Positional encoding for spatial awareness
        self.pos_encoding = nn.Parameter(
            torch.randn(1, spatial_dim * spatial_dim, feature_dim) * 0.02
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable query tokens for epistemic-relevant features
        self.epistemic_queries = nn.Parameter(
            torch.randn(1, 16, feature_dim) * 0.02  # 16 query tokens
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(16 * feature_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature maps
        Returns:
            (B, 256) aggregated features
        """
        B, C, H, W = x.shape
        
        # Reshape to sequence format
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :H*W, :]
        
        # Expand queries for batch
        queries = self.epistemic_queries.expand(B, -1, -1)
        
        # Apply multi-head attention
        attended, attention_weights = self.attention(
            query=queries,
            key=x,
            value=x,
            need_weights=True
        )
        
        # Flatten and project
        attended_flat = attended.reshape(B, -1)
        output = self.output_proj(attended_flat)
        
        return output, attention_weights
    
def visualize_reconstruction(
    agent,
    observations: torch.Tensor,
    frame_idx: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_samples: int = 4,
    visualization_mode: str = "recent"
) -> float:
    """
    Visualize observation reconstruction through the diffusion latent space.
    Properly handles frame-stacked inputs with single-frame decoder outputs.
    
    Args:
        agent: The agent to visualize
        observations: Batch of observations (may be frame-stacked)
        save_path: Where to save the visualization
        max_samples: Maximum number of samples to visualize
        visualization_mode: How to visualize frame stacks
    """
    device = agent.device if hasattr(agent, 'device') else agent.active_inference.device
    
    # Ensure we're in eval mode
    was_training = agent.active_inference.training
    agent.active_inference.eval()
    
    with torch.no_grad():
        # Move observations to device
        if observations.device != device:
            observations = observations.to(device)
        if frame_idx is not None and not torch.is_tensor(frame_idx):
            frame_idx = torch.as_tensor(frame_idx, device=device)
        if actions is not None and actions.device != device:
            actions = actions.to(device)

        # For pixel observations, we need to encode them first
        if hasattr(agent, 'encoder') and agent.config.pixel_observation:
            # Encode pixel observations to features
            encoded_obs = agent.encode_observation(observations[:max_samples])
        else:
            # For state observations, use them directly
            encoded_obs = observations[:max_samples]
        
        # Generate latents via diffusion
        belief_info = agent.active_inference.update_belief_via_diffusion(encoded_obs, 
                                                                         frame_idx=frame_idx[:max_samples] if frame_idx is not None else None,
                                                                         actions=actions[:max_samples] if actions is not None else None)
        latents = belief_info['latent']
        
        # Decode latents back to observation space
        reconstructed_obs = agent.active_inference.decode_observation(latents)
        
        # Debug prints to understand shapes
        print(f"Original observations shape: {observations[:max_samples].shape}")
        print(f"Encoded observations shape: {encoded_obs.shape}")
        print(f"Latents shape: {latents.shape}")
        print(f"Reconstructed observations shape: {reconstructed_obs.shape}")
        
        # For pixel observations, handle visualization
        if agent.config.pixel_observation:
            if save_path:
                # Determine number of samples to visualize
                n_samples = min(max_samples, observations.shape[0])
                
                # Process observations and reconstructions
                orig_obs = observations[:n_samples].cpu().numpy()
                recon_obs = reconstructed_obs[:n_samples].cpu().numpy()
                
                # Check if we're dealing with frame-stacked input and single-frame output
                is_frame_stacked_input = (orig_obs.ndim == 4 and orig_obs.shape[1] > 3) or \
                                       (orig_obs.ndim == 5)
                # supports NCHW (C==3) or NHWC (last dim == 3)
                is_single_frame_output = (recon_obs.ndim == 4 and (recon_obs.shape[1] == 3 or recon_obs.shape[-1] == 3))

                if is_frame_stacked_input and is_single_frame_output:
                    print("Detected frame-stacked input with single-frame decoder output")
                    # This is the common case - we'll compare the most recent frame
                    # with the reconstruction
                    create_mixed_reconstruction_plot(
                        orig_obs, recon_obs, save_path, visualization_mode
                    )
                else:
                    # Standard case - matching dimensions
                    create_standard_reconstruction_plot(
                        orig_obs, recon_obs, save_path, visualization_mode
                    )
            
            # Compute reconstruction error in feature space
            predicted_features = agent.active_inference.decode_observation(latents, decode_to_pixels=False)
            recon_error = F.mse_loss(predicted_features, encoded_obs).item()
                
        else:
            # State reconstruction (unchanged)
            recon_error = F.mse_loss(reconstructed_obs, observations[:max_samples]).item()
            
            if save_path:
                # State visualization code remains the same
                pass
    
    # Restore training mode
    if was_training:
        agent.active_inference.train()
    
    return recon_error


def create_mixed_reconstruction_plot(
    frame_stacked_obs: np.ndarray,
    single_frame_recon: np.ndarray,
    save_path: str,
    mode: str = "recent"  # use "separate" here to show 3 distinct panels
):
    """
    Create plot comparing frame-stacked observations with single-frame reconstructions.
    For C=9 inputs, we reshape to (3,3,H,W) and either stitch or show separate panels.
    """
    n_samples = frame_stacked_obs.shape[0]
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))
    if n_samples == 1:
        axes = axes.reshape(3, 1)

    for i in range(n_samples):
        orig = frame_stacked_obs[i]
        recon = single_frame_recon[i]

        # --- extract frames ---
        if orig.ndim == 4:  # (frames, channels, H, W)
            frames = orig
        elif orig.ndim == 3 and orig.shape[0] > 3 and (orig.shape[0] % 3 == 0):
            n_frames = orig.shape[0] // 3
            h, w = orig.shape[1:]
            frames = orig.reshape(n_frames, 3, h, w)  # (F,3,H,W) e.g., (3,3,H,W)
        else:
            frames = np.expand_dims(orig, 0) if orig.ndim == 3 else orig  # (1,3,H,W) or (1,H,W,3)

        # --- Row 1: show all frames ---
        ax = axes[0, i]
        ax.axis('off')
        ax.set_title(f'Frame Stack {i}')
        F = frames.shape[0]

        if F > 1 and mode in ("separate", "separate_frames"):
            # draw 3 separate panels (no concatenation, uses ALL frames)
            left_margin = 0.02
            right_margin = 0.02
            w_frac = (1.0 - left_margin - right_margin) / F
            for k in range(F):
                fr = frames[k]
                # CHW -> HWC if needed
                fr = np.transpose(fr, (1, 2, 0)) if (fr.ndim == 3 and fr.shape[0] == 3) else fr
                if fr.ndim == 2:
                    fr = fr[..., None]
                if fr.shape[-1] not in (1, 3):
                    fr = fr[..., :3]
                if fr.max() > 1.0:
                    fr = fr / 255.0
                fr = np.clip(fr, 0, 1)
                x0 = left_margin + k * w_frac
                iax = inset_axes(ax, width=f"{w_frac*100:.3f}%", height="100%",
                                 bbox_to_anchor=(x0, 0.0, w_frac, 1.0),
                                 bbox_transform=ax.transAxes, borderpad=0)
                iax.imshow(fr if fr.shape[-1] == 3 else np.repeat(fr, 3, axis=2))
                iax.axis('off')
        else:
            # current behavior: stitch frames horizontally into one image
            frame_grid = []
            for k in range(F):
                fr = frames[k]
                fr = np.transpose(fr, (1, 2, 0)) if (fr.ndim == 3 and fr.shape[0] == 3) else fr
                if fr.ndim == 2:
                    fr = fr[..., None]
                if fr.shape[-1] not in (1, 3):
                    fr = fr[..., :3]
                if fr.max() > 1.0:
                    fr = fr / 255.0
                frame_grid.append(np.clip(fr, 0, 1))
            grid = np.concatenate(frame_grid, axis=1)
            ax.imshow(grid)

        # --- Row 2: most recent frame (target) ---
        most_recent_vis = process_single_frame(frames[-1])
        axes[1, i].imshow(most_recent_vis); axes[1, i].axis('off'); axes[1, i].set_title(f'Target Frame {i}')

        # --- Row 3: reconstruction ---
        recon_vis = process_single_frame(recon)
        axes[2, i].imshow(recon_vis); axes[2, i].axis('off'); axes[2, i].set_title(f'Reconstruction {i}')

        # PSNR (optional)
        mse = np.mean((most_recent_vis - recon_vis) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        axes[2, i].text(0.5, -0.1, f'PSNR: {psnr:.1f}dB',
                        transform=axes[2, i].transAxes, ha='center', fontsize=8)

    # Row labels
    fig.text(0.02, 0.75, 'Input\nStack', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.5,  'Target\nFrame', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.25, 'Recon', rotation=90, va='center', fontsize=12, weight='bold')
    plt.tight_layout(); plt.subplots_adjust(left=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved frame-stack aware reconstruction visualization to {save_path}")

def create_standard_reconstruction_plot(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
    save_path: str,
    mode: str = "recent"
):
    """Standard reconstruction plot when dimensions match"""
    n_samples = original_images.shape[0]
    
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_samples):
        # Process observations
        orig = process_observation_for_display(original_images[i], mode)
        recon = process_observation_for_display(reconstructed_images[i], mode)
        
        # Original
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f'Original {i}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon)
        axes[1, i].set_title(f'Reconstructed {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved standard reconstruction visualization to {save_path}")


def process_observation_for_display(obs: np.ndarray, mode: str) -> np.ndarray:
    """Process any observation format for display."""
    # If it's a frame stack (either 4D or CHW with C>3 and divisible by 3),
    # render an RGB frame from it (e.g., most recent or grid).
    if obs.ndim == 4:
        return process_frame_stack(obs, mode)
    if obs.ndim == 3 and obs.shape[0] > 3 and (obs.shape[0] % 3 == 0):
        # CHW where C is 3 * num_frames
        return process_frame_stack(obs, mode)
    # Otherwise treat as a single frame
    return process_single_frame(obs)


def process_single_frame(frame: np.ndarray) -> np.ndarray:
    """Process a single frame for visualization"""
    # Handle different formats
    if frame.ndim == 3 and frame.shape[0] in [1, 3]:  # (C, H, W)
        frame = np.transpose(frame, (1, 2, 0))
    elif frame.ndim == 2:  # (H, W)
        frame = np.expand_dims(frame, axis=-1) #(H, W, 1)
    # Handle CHW with C > 3 (e.g., 9): take the most recent RGB triplet
    elif frame.ndim == 3 and frame.shape[0] > 3 and (frame.shape[0] % 3 == 0):
        n_frames = frame.shape[0] // 3
        c_last = 3 * (n_frames - 1)
        last_rgb = frame[c_last:c_last + 3, ...]  # (3, H, W)
        frame = np.transpose(last_rgb, (1, 2, 0))  # -> (H, W, 3)
    # If it's already HWC but channels != 1/3, fall back to first 3 channels
    elif frame.ndim == 3 and frame.shape[-1] not in (1, 3):
        frame = frame[..., :3]

    # Handle grayscale
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=2)
    
    # Ensure proper range
    if frame.max() > 1.0:
        frame = frame / 255.0
    
    return np.clip(frame, 0, 1)


def process_frame_stack(frames: np.ndarray, mode: str = "recent") -> np.ndarray:
    """Process frame-stacked observations for visualization"""
    # First, determine the format
    if frames.shape[0] > 4:  # Likely concatenated channels
        # Assume RGB frames concatenated
        n_channels = 3
        n_frames = frames.shape[0] // n_channels
        h, w = frames.shape[-2:]
        frames = frames.reshape(n_frames, n_channels, h, w)
    
    # Now frames should be (num_frames, channels, H, W)
    num_frames = frames.shape[0]
    
    if mode == "recent":
        # Use the most recent frame
        frame = frames[-1]
    elif mode == "grid":
        # Create a grid showing all frames
        grid_frames = []
        for i in range(num_frames):
            f = process_single_frame(frames[i])
            grid_frames.append(f)
        # Concatenate horizontally
        return np.concatenate(grid_frames, axis=1)
    elif mode == "blend":
        # Average frames with recency weighting
        weights = np.linspace(0.1, 1.0, num_frames)
        weights = weights / weights.sum()
        blended = np.zeros_like(frames[0], dtype=np.float32)
        for i, w in enumerate(weights):
            blended += w * frames[i]
        frame = blended
    else:
        frame = frames[-1]  # Default to most recent
    
    return process_single_frame(frame)

# --- Dreamer-style symlog / symexp and two-hot utils ---
def symlog(x):
    return x.sign() * (x.abs() + 1.0).log()

def symexp(x):
    return x.sign() * (x.abs().exp() - 1.0)

def make_symlog_bins(num_bins=255, lo=-20.0, hi=20.0, device=None, dtype=None):
    bins_symlog = torch.linspace(lo, hi, num_bins, device=device, dtype=dtype)
    bins_real = symexp(bins_symlog)
    return bins_symlog, bins_real

def twohot_encode(x_symlog, bins_symlog):
    B = bins_symlog.shape[0]
    lo, hi = bins_symlog[0], bins_symlog[-1]
    pos = (x_symlog - lo) / (hi - lo) * (B - 1)
    pos = pos.clamp(0, B - 1 - torch.finfo(x_symlog.dtype).eps)
    idx0 = pos.floor().long()
    idx1 = (idx0 + 1).clamp_max(B - 1)
    w1 = (pos - idx0.float())
    w0 = 1.0 - w1
    # build soft labels
    shape = x_symlog.shape + (B,)
    target = torch.zeros(shape, device=bins_symlog.device, dtype=torch.float32)
    # scatter weights into target
    target.scatter_(-1, idx0.unsqueeze(-1), w0.unsqueeze(-1))
    target.scatter_add_(-1, idx1.unsqueeze(-1), w1.unsqueeze(-1))
    return target

def categorical_ce_with_soft_targets(logits, soft_targets):
    logp = torch.log_softmax(logits, dim=-1)
    return -(soft_targets * logp).sum(dim=-1)
