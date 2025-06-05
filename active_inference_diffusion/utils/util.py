from ..agents.pixel_agent import DiffusionPixelAgent
from ..agents.state_agent import DiffusionStateAgent
import torch
import matplotlib.pyplot as plt

def visualize_reconstruction(agent, observations, save_path="reconstruction.png"):
    """
    Visualize original vs reconstructed observations for debugging.
    Works for both state and pixel observations.
    """
    
    
    with torch.no_grad():
        # Process observations
        if isinstance(agent, DiffusionPixelAgent):
            # For pixel agent, encode first
            encoded = agent.encode_observation(observations)
            belief_info = agent.active_inference.update_belief_via_diffusion(encoded)
        elif isinstance(agent, DiffusionStateAgent):
            # For state agent
            belief_info = agent.active_inference.update_belief_via_diffusion(observations)
        
        latents = belief_info['latent']
        
        # Reconstruct
        reconstructed = agent.active_inference.decode_observation(latents)
        
        # Handle different observation types
        if len(observations.shape) == 4:  # RGB: (B, C, H, W)
            # Take first 4 samples
            n_samples = min(4, observations.shape[0])
            fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*3, 6))
            
            for i in range(n_samples):
                # Original
                img = observations[i].cpu().permute(1, 2, 0).numpy()
                if img.max() <= 1.0:
                    img = img  # Already normalized
                else:
                    img = img / 255.0
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"Original {i}")
                axes[0, i].axis('off')
                
                # Reconstructed 
                recon = reconstructed[i].cpu().permute(1, 2, 0).numpy()
                axes[1, i].imshow(recon)
                axes[1, i].set_title(f"Reconstructed {i}")
                axes[1, i].axis('off')
                
        else:  # State observations
            # Plot state dimensions
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            x = range(observations.shape[-1])
            
            # Plot first few samples
            n_samples = min(3, observations.shape[0])
            for i in range(n_samples):
                ax.plot(x, observations[i].cpu(), 'o-', label=f'Original {i}', alpha=0.7)
                ax.plot(x, reconstructed[i].cpu(), 's--', label=f'Reconstructed {i}', alpha=0.7)
            
            ax.set_xlabel('State Dimension')
            ax.set_ylabel('Value')
            ax.set_title('State Reconstruction Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Reconstruction Error: {belief_info["reconstruction_error"]:.4f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return belief_info["reconstruction_error"].item()