import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import wandb
from pathlib import Path
import argparse
import yaml
from typing import Dict, Optional
# Import from our package
from active_inference_diffusion.agents import StateBasedAgent
from active_inference_diffusion.envs.wrappers import make_pixel_mujoco
from examples.configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)
from active_inference_diffusion.utils.logger import Logger
from active_inference_diffusion.utils.training import save_checkpoint, load_checkpoint, evaluate_agent    
from active_inference_diffusion.utils.training import create_video
from active_inference_diffusion.utils.training import plot_training_curves

def train_state_based(env_name: str, config_path: Optional[str] = None, resume: bool = False):
    """Train state-based Active Inference agent"""
    
    # Load or create config
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        ai_config = ActiveInferenceConfig(**config_dict.get('active_inference', {}))
        train_config = TrainingConfig(**config_dict.get('training', {}))
    else:
        ai_config = ActiveInferenceConfig(env_name=env_name)
        train_config = TrainingConfig()
        
    # Create environment
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Create agent
    agent = StateBasedAgent(env, ai_config, train_config)
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
        experiment_name=f"{env_name}_state_ai",
        config={**ai_config.__dict__, **train_config.__dict__}
    )
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in tqdm(range(start_step, train_config.total_timesteps), desc="Training"):
        # Act
        action, info = agent.act(obs, deterministic=False)
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Update state
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
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
        if step >= train_config.learning_starts and step % train_config.train_frequency == 0:
            for _ in range(train_config.gradient_steps):
                train_metrics = agent.train_step()
                
            if step % train_config.log_frequency == 0:
                logger.log(train_metrics, step=step)
                
        # Update exploration
        agent.update_exploration()
        
        # Evaluate
        if step % train_config.eval_frequency == 0 and step > 0:
            eval_metrics = evaluate_agent(agent, eval_env, train_config.num_eval_episodes)
            logger.log(eval_metrics, step=step)
            print(f"Step {step}: Eval Reward = {eval_metrics['eval/mean_reward']:.2f}")
            
        # Save checkpoint
        if step % train_config.save_frequency == 0 and step > 0:
            save_checkpoint(agent, f"checkpoint_{step}.pt")
            
    env.close()
    eval_env.close()
    logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                       help='Environment name')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')  # Add this
    
    args = parser.parse_args()
     # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_state_based(args.env, args.config, resume=args.resume)
    create_video(agent, args.env, "halfcheetah_final.mp4", num_episodes=5)
    plot_training_curves(
    "logs/experiment.jsonl", 
    "training_progress.png",
    metrics=['episode/reward', 'free_energy', 'belief/entropy']
   )