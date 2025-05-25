"""
Training utilities for Active Inference + Diffusion
Shared functions for training, evaluation, and checkpointing
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Optional, Any
import json

def evaluate_agent(
    agent: Any, 
    env: gym.Env, 
    num_episodes: int,
    max_episode_length: int = 1000
) -> Dict[str, float]:
    """
    Evaluate agent performance
    
    Args:
        agent: The agent to evaluate
        env: Environment to evaluate in
        num_episodes: Number of evaluation episodes
        max_episode_length: Maximum steps per episode
        
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    lengths = []
    
    # Set to eval mode
    agent.active_inference.eval()
    
    for _ in range(num_episodes):  # Fixed: was "for * in range(num*episodes)"
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_episode_length:
            # Get action (deterministic for evaluation)
            action, _ = agent.act(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    # Back to train mode
    agent.active_inference.train()
    
    return {
        'eval/mean_reward': np.mean(rewards),
        'eval/std_reward': np.std(rewards),
        'eval/min_reward': np.min(rewards),
        'eval/max_reward': np.max(rewards),
        'eval/mean_length': np.mean(lengths),
        'eval/std_length': np.std(lengths)
    }


def save_checkpoint(
    agent: Any, 
    filename: str,
    checkpoint_dir: str = "checkpoints",
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save training checkpoint
    
    Args:
        agent: The agent to save
        filename: Checkpoint filename
        checkpoint_dir: Directory to save checkpoints
        additional_info: Any additional info to save
    """
    checkpoint = {
        # Model states
        'active_inference_state': agent.active_inference.state_dict(),
        
        # Optimizer states
        'score_optimizer': agent.score_optimizer.state_dict(),
        'policy_optimizer': agent.policy_optimizer.state_dict(),
        'value_optimizer': agent.value_optimizer.state_dict(),
        'dynamics_optimizer': agent.dynamics_optimizer.state_dict(),
        
        # Training state
        'total_steps': agent.total_steps,
        'episode_count': agent.episode_count,
        'exploration_noise': agent.exploration_noise,
        
        # Configuration
        'config': agent.config,
        'training_config': agent.training_config,
    }
    
    # Add any additional info
    if additional_info:
        checkpoint.update(additional_info)
    
    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # Save checkpoint
    full_path = checkpoint_path / filename
    torch.save(checkpoint, full_path)
    print(f"✅ Saved checkpoint: {full_path}")
    
    # Also save a 'latest' checkpoint for easy resuming
    latest_path = checkpoint_path / "latest.pt"
    torch.save(checkpoint, latest_path)


def load_checkpoint(
    agent: Any, 
    filename: str,
    checkpoint_dir: str = "checkpoints",
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load training checkpoint
    
    Args:
        agent: The agent to load into
        filename: Checkpoint filename
        checkpoint_dir: Directory containing checkpoints
        strict: Whether to strictly enforce state dict matching
        
    Returns:
        The loaded checkpoint dictionary
    """
    # Build path
    checkpoint_path = Path(checkpoint_dir) / filename
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    # Load model state
    agent.active_inference.load_state_dict(
        checkpoint['active_inference_state'], 
        strict=strict
    )
    
    # Load optimizer states
    agent.score_optimizer.load_state_dict(checkpoint['score_optimizer'])
    agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
    agent.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
    agent.dynamics_optimizer.load_state_dict(checkpoint['dynamics_optimizer'])
    
    # Load training state
    agent.total_steps = checkpoint['total_steps']
    agent.episode_count = checkpoint['episode_count']
    agent.exploration_noise = checkpoint['exploration_noise']
    
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    print(f"   Resuming from step {agent.total_steps}, episode {agent.episode_count}")
    
    return checkpoint

"""
Video creation and plotting utilities
"""




def create_video(
    agent: Any,
    env: gym.Env,
    filename: str = "evaluation.mp4",
    num_episodes: int = 1,
    fps: int = 30,
    video_folder: str = "videos"
):
    """
    Create video of agent performance
    
    Args:
        agent: The trained agent to record
        env: Gymnasium environment (will be wrapped with RecordVideo)
        filename: Output video filename
        num_episodes: Number of episodes to record
        fps: Frames per second for video
        video_folder: Directory to save videos
        
    Example:
        >>> from active_inference_diffusion.utils.training import create_video
        >>> create_video(agent, env, "halfcheetah_trained.mp4", num_episodes=3)
    """
    from gymnasium.wrappers import RecordVideo
    
    # Create video directory
    Path(video_folder).mkdir(exist_ok=True)
    
    # Wrap environment with video recorder
    video_env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=filename.split('.')[0],
        episode_trigger=lambda episode_id: True,  # Record all episodes
        disable_logger=True
    )
    
    # Set agent to evaluation mode
    agent.active_inference.eval()
    
    print(f"Recording {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = video_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action deterministically
            action, _ = agent.act(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    # Close environment
    video_env.close()
    
    # Back to training mode
    agent.active_inference.train()
    
    print(f"Video saved to: {video_folder}/")


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
    metrics: List[str] = ['episode/reward', 'free_energy', 'expected_free_energy'],
    window_size: int = 100,
    figsize: tuple = (12, 8)
):
    """
    Plot training curves from log file
    
    Args:
        log_file: Path to JSONL log file created by Logger
        save_path: Where to save the plot (if None, displays plot)
        metrics: List of metrics to plot
        window_size: Window size for moving average smoothing
        figsize: Figure size (width, height)
        
    Example:
        >>> from active_inference_diffusion.utils.training import plot_training_curves
        >>> plot_training_curves("logs/experiment.jsonl", "training_curves.png")
    """
    # Check if log file exists
    if not Path(log_file).exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    # Load data from JSONL file
    data = {metric: [] for metric in metrics}
    steps = []
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                if 'step' in entry:
                    steps.append(entry['step'])
                    for metric in metrics:
                        if metric in entry:
                            data[metric].append(entry[metric])
                        else:
                            # Handle missing data
                            if len(data[metric]) > 0:
                                data[metric].append(data[metric][-1])
                            else:
                                data[metric].append(0)
            except json.JSONDecodeError:
                continue
    
    # Create figure with subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = data[metric]
        
        if len(values) > 0:
            # Plot raw data with transparency
            ax.plot(steps[:len(values)], values, alpha=0.3, color='blue', label='Raw')
            
            # Calculate and plot moving average
            if len(values) > window_size:
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                avg_steps = steps[window_size-1:len(moving_avg)+window_size-1]
                ax.plot(avg_steps, moving_avg, color='red', linewidth=2, label=f'MA-{window_size}')
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric.replace('/', ' ').title())
            ax.set_title(f'{metric} over Training')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title(f'{metric} (No Data)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_belief_evolution(
    agent: Any,
    save_path: Optional[str] = None,
    dimensions_to_plot: int = 5
):
    """
    Plot the evolution of belief dynamics over time
    
    Args:
        agent: Agent with belief dynamics
        save_path: Where to save the plot
        dimensions_to_plot: Number of latent dimensions to visualize
        
    Example:
        >>> plot_belief_evolution(agent, "belief_evolution.png")
    """
    if not hasattr(agent.active_inference, 'belief_dynamics'):
        print("Agent does not have belief dynamics")
        return
        
    belief_dynamics = agent.active_inference.belief_dynamics
    history = belief_dynamics.history
    
    if not history['means']:
        print("No belief history to plot")
        return
    
    # Convert to numpy
    means = torch.stack(history['means']).numpy()
    covs = torch.stack(history['covariances']).numpy()
    entropies = torch.stack(history['entropies']).numpy()
    free_energies = np.array(history['free_energies'])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mean evolution
    ax = axes[0, 0]
    n_dims = min(dimensions_to_plot, means.shape[1])
    for i in range(n_dims):
        ax.plot(means[:, i], label=f'Dim {i}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Belief Mean')
    ax.set_title('Belief Mean Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty (trace of covariance)
    ax = axes[0, 1]
    uncertainties = [np.trace(cov) for cov in covs]
    ax.plot(uncertainties)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Uncertainty (Trace of Covariance)')
    ax.set_title('Belief Uncertainty Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Entropy
    ax = axes[1, 0]
    ax.plot(entropies)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Entropy')
    ax.set_title('Belief Entropy Evolution')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Free Energy
    ax = axes[1, 1]
    ax.plot(free_energies)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Free Energy')
    ax.set_title('Free Energy Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Belief evolution plot saved to: {save_path}")
    else:
        plt.show()
        
    plt.close()