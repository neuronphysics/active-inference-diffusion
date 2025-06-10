import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union
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
    agent,  # Type annotation removed to avoid circular import
    observations: torch.Tensor,
    save_path: Optional[str] = None,
    max_samples: int = 4
) -> float:
    """
    Visualize observation reconstruction through the diffusion latent space.
    Works for both pixel and state observations.
    
    This function demonstrates how observations are encoded to latents
    and then decoded back, which is crucial for computing epistemic value.
    """
    device = observations.device
    
    # Ensure we're in eval mode
    was_training = agent.active_inference.training
    agent.active_inference.eval()
    
    with torch.no_grad():
        # Move observations to device
        if observations.device != device:
            observations = observations.to(device)
            
        # For pixel observations, we need to encode them first
        if hasattr(agent, 'encoder') and agent.config.pixel_observation:
            # Encode pixel observations to features
            encoded_obs = agent.encode_observation(observations[:max_samples])
        else:
            # For state observations, use them directly
            encoded_obs = observations[:max_samples]
        
        # Generate latents via diffusion
        belief_info = agent.active_inference.update_belief_via_diffusion(encoded_obs)
        latents = belief_info['latent']
        
        # Decode latents back to observation space
        reconstructed_obs = agent.active_inference.decode_observation(latents)
        
        # For pixel observations, we need to handle the output differently
        if agent.config.pixel_observation:
            # Pixel reconstruction
            if save_path:
                fig, axes = plt.subplots(2, max_samples, figsize=(max_samples * 3, 6))
                
                for i in range(min(max_samples, observations.shape[0])):
                    # Original observation
                    if observations.shape[1] > 3:  # Frame stacked
                        # Show only the most recent frame
                        orig = observations[i, -3:].cpu().numpy()
                    else:
                        orig = observations[i].cpu().numpy()
                    
                    # Handle channel format
                    if orig.shape[0] == 3:  # (C, H, W)
                        orig = np.transpose(orig, (1, 2, 0))
                    
                    # Reconstructed observation
                    recon = reconstructed_obs[i].cpu().numpy()
                    if recon.shape[0] == 3:  # (C, H, W)
                        recon = np.transpose(recon, (1, 2, 0))
                    
                    # Ensure values are in [0, 1]
                    orig = np.clip(orig, 0, 1)
                    recon = np.clip(recon, 0, 1)
                    
                    # Plot
                    axes[0, i].imshow(orig)
                    axes[0, i].set_title(f'Original {i}')
                    axes[0, i].axis('off')
                    
                    axes[1, i].imshow(recon)
                    axes[1, i].set_title(f'Reconstructed {i}')
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Compute reconstruction error
            if observations.shape == reconstructed_obs.shape:
                recon_error = F.mse_loss(reconstructed_obs, observations[:max_samples]).item()
            else:
                # If shapes don't match (e.g., frame stacking), compare encoded features
                recon_error = F.mse_loss(reconstructed_obs, encoded_obs).item()
                
        else:
            # State reconstruction
            recon_error = F.mse_loss(reconstructed_obs, observations[:max_samples]).item()
            
            if save_path:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                # Plot first few dimensions
                dims_to_plot = min(5, observations.shape[1])
                x = np.arange(dims_to_plot)
                
                for i in range(min(max_samples, observations.shape[0])):
                    orig = observations[i, :dims_to_plot].cpu().numpy()
                    recon = reconstructed_obs[i, :dims_to_plot].cpu().numpy()
                    
                    offset = i * 0.2
                    ax.plot(x, orig + offset, 'o-', label=f'Original {i}', alpha=0.7)
                    ax.plot(x, recon + offset, 's--', label=f'Recon {i}', alpha=0.7)
                
                ax.set_xlabel('State Dimension')
                ax.set_ylabel('Value (offset for clarity)')
                ax.set_title('State Reconstruction Quality')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    # Restore training mode
    if was_training:
        agent.active_inference.train()
    
    return recon_error
