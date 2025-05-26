import os
import sys
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
import yaml
from typing import Dict, Optional

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our package
from active_inference_diffusion.agents import PixelBasedAgent
from active_inference_diffusion.envs.wrappers import make_pixel_mujoco
from active_inference_diffusion.configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig,
    DiffusionConfig,
    BeliefDynamicsConfig
)
from active_inference_diffusion.utils.logger import Logger
from active_inference_diffusion.utils.training import (
    save_checkpoint,
    load_checkpoint,
    evaluate_agent,
    create_video,
    plot_training_curves
)


def load_config(config_path: str) -> tuple:
    """Load and parse YAML config"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Parse nested configs
    ai_config_dict = config_dict.get('active_inference', {})
    
    # Create nested config objects
    if 'diffusion' in ai_config_dict:
        ai_config_dict['diffusion'] = DiffusionConfig(**ai_config_dict['diffusion'])
    
    if 'belief_dynamics' in ai_config_dict:
        ai_config_dict['belief_dynamics'] = BeliefDynamicsConfig(**ai_config_dict['belief_dynamics'])
    
    # Create main configs
    ai_config = ActiveInferenceConfig(**ai_config_dict)
    train_config = TrainingConfig(**config_dict.get('training', {}))
    pixel_config = PixelObservationConfig(**config_dict.get('pixel', {}))
    
    return ai_config, train_config, pixel_config


def train_pixel_based(
    env_name: str, 
    config_path: Optional[str] = None,
    resume: bool = False,
    seed: int = 0
):
    """Train pixel-based Active Inference agent"""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load or create config
    if config_path:
        ai_config, train_config, pixel_config = load_config(config_path)
        # Override env name if specified
        ai_config.env_name = env_name
    else:
        ai_config = ActiveInferenceConfig(env_name=env_name)
        train_config = TrainingConfig()
        pixel_config = PixelObservationConfig()
    
    # Create pixel environments
    env = make_pixel_mujoco(
        env_id=env_name,
        width=pixel_config.image_shape[1],
        height=pixel_config.image_shape[2],
        frame_stack=pixel_config.frame_stack,
        camera_name="track"  # Default camera
    )
    
    eval_env = make_pixel_mujoco(
        env_id=env_name,
        width=pixel_config.image_shape[1],
        height=pixel_config.image_shape[2],
        frame_stack=pixel_config.frame_stack,
        camera_name="track"
    )
    
    # Set seeds for environments
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1000)
    
    # Create agent
    agent = PixelBasedAgent(env, ai_config, train_config, pixel_config)
    
    # Resume from checkpoint if requested
    start_step = 0
    if resume:
        try:
            checkpoint = load_checkpoint(agent, "latest.pt")
            start_step = agent.total_steps
            print(f"Resuming training from step {start_step}")
        except FileNotFoundError:
            print("No checkpoint found, starting fresh")
    
    # Setup logging
    logger = Logger(
        use_wandb=train_config.use_wandb,
        project_name=train_config.project_name,
        experiment_name=f"{env_name}_pixel_ai_seed{seed}",
        config={
            **ai_config.__dict__,
            **train_config.__dict__,
            **pixel_config.__dict__,
            'seed': seed
        }
    )
    
    # Training loop
    train_loop(agent, env, eval_env, train_config, logger, start_step)
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate_agent(agent, eval_env, train_config.num_eval_episodes * 2)
    logger.log(final_metrics, step=train_config.total_timesteps)
    print(f"Final Eval Reward: {final_metrics['eval/mean_reward']:.2f}")
    
    # Create video with best model
    try:
        load_checkpoint(agent, "best.pt")
        video_path = f"videos/{env_name}_pixel_best.mp4"
        create_video(agent, eval_env, video_path, num_episodes=5)
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Could not create video: {e}")
    
    # Plot training curves
    try:
        plot_training_curves(
            logger.log_path,
            f"plots/{env_name}_pixel_training_curves.png",
            metrics=['episode/reward', 'free_energy', 'belief/entropy', 'contrastive_loss']
        )
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    env.close()
    eval_env.close()
    logger.finish()


def train_loop(agent, env, eval_env, config, logger, start_step=0):
    """Generic training loop"""
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    best_eval_reward = -float('inf')
    
    for step in tqdm(range(start_step, config.total_timesteps), desc="Training"):
        # Act
        action, info = agent.act(obs, deterministic=False)
        
        # Log belief dynamics info periodically
        if step % 100 == 0:
            logger.log({
                'belief/entropy': info.get('belief_entropy', 0),
                'belief/expected_free_energy': info.get('expected_free_energy', 0),
                'exploration/noise': agent.exploration_noise
            }, step=step)
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Update state
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        agent.total_steps = step
        
        # Reset if done
        if done:
            logger.log({
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'episode/count': agent.episode_count
            }, step=step)
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            agent.episode_count += 1
        
        # Train
        if step >= config.learning_starts and step % config.train_frequency == 0:
            for _ in range(config.gradient_steps):
                train_metrics = agent.train_step()
            
            if step % config.log_frequency == 0:
                logger.log(train_metrics, step=step)
        
        # Update exploration
        agent.update_exploration()
        
        # Evaluate
        if step % config.eval_frequency == 0 and step > 0:
            eval_metrics = evaluate_agent(agent, eval_env, config.num_eval_episodes)
            logger.log(eval_metrics, step=step)
            
            eval_reward = eval_metrics['eval/mean_reward']
            print(f"Step {step}: Eval Reward = {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_checkpoint(agent, "best.pt")
                print(f"New best model saved! Reward: {eval_reward:.2f}")
        
        # Save checkpoint
        if step % config.save_frequency == 0 and step > 0:
            save_checkpoint(agent, f"checkpoint_{step}.pt")
            save_checkpoint(agent, "latest.pt")  # Always keep latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                        help='Environment name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    
    args = parser.parse_args()
    
    # Use default config if not specified
    if args.config is None:
        config_path = f"configs/halfcheetah_pixel.yaml"
        if os.path.exists(config_path):
            args.config = config_path
    
    train_pixel_based(args.env, args.config, resume=args.resume, seed=args.seed)