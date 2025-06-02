"""
Training script for Diffusion Active Inference on MuJoCo
"""

import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any

from active_inference_diffusion.agents import DiffusionStateAgent, DiffusionPixelAgent
from active_inference_diffusion.configs.config import (
    ActiveInferenceConfig, 
    TrainingConfig,
    DiffusionConfig,
    PixelObservationConfig
)
from active_inference_diffusion.utils.logger import Logger
from active_inference_diffusion.utils.training import (
    evaluate_agent,
    save_checkpoint,
    create_video,
    plot_training_curves
)
from active_inference_diffusion.envs.wrappers import NormalizeObservation, ActionRepeat
from active_inference_diffusion.envs.pixel_wrappers import make_pixel_mujoco


def setup_environment(
    env_name: str,
    use_pixels: bool = False,
    seed: int = 0
) -> gym.Env:
    """Setup MuJoCo environment with appropriate wrappers"""
    
    if use_pixels:
        # Pixel-based environment
        env = make_pixel_mujoco(
            env_name,
            width=84,
            height=84,
            frame_stack=3,
            action_repeat=2,
            seed=seed
        )
    else:
        # State-based environment
        env = gym.make(env_name)
        env = NormalizeObservation(env)
        env = ActionRepeat(env, repeat=2)
        
        # Set seeds
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    return env


def train_diffusion_active_inference(
    env_name: str = "HalfCheetah-v4",
    use_pixels: bool = False,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Main training function"""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = setup_environment(env_name, use_pixels, seed)
    eval_env = setup_environment(env_name, use_pixels, seed + 100)
    
    # Create configurations
    config = ActiveInferenceConfig(
        env_name=env_name,
        latent_dim=50,
        hidden_dim=256,
        learning_rate=3e-4,
        batch_size=256,
        efe_horizon=5,
        epistemic_weight=0.1,
        pragmatic_weight=1.0,
        consistency_weight=0.1,
        kl_weight=0.1,
        diffusion_weight=1.0,
        pixel_observation=use_pixels,
        device=device
    )
    
    # Enhanced diffusion config
    config.diffusion = DiffusionConfig(
        num_diffusion_steps=50,  # Fewer steps for faster inference
        beta_schedule="cosine",
        beta_start=1e-4,
        beta_end=0.02
    )
    
    training_config = TrainingConfig(
        total_timesteps=total_timesteps,
        eval_frequency=10_000,
        save_frequency=50_000,
        log_frequency=1_000,
        buffer_size=100_000,
        learning_starts=10_000,
        gradient_steps=2,
        exploration_noise=0.1,
        exploration_decay=0.999,
    )
    
    # Create agent
    if use_pixels:
        pixel_config = PixelObservationConfig()
        agent = DiffusionPixelAgent(
            env, config, training_config, pixel_config
        )
    else:
        agent = DiffusionStateAgent(env, config, training_config)
        
    # Create logger
    logger = Logger(
        use_wandb=True,
        project_name="diffusion-active-inference-mujoco",
        experiment_name=f"{env_name}_{'pixels' if use_pixels else 'states'}_seed{seed}",
        config={
            **config.__dict__,
            **training_config.__dict__
        }
    )
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in range(total_timesteps):
        # Select action
        action, info = agent.act(obs, deterministic=False)
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Update counters
        episode_reward += reward
        episode_length += 1
        agent.total_steps += 1
        train_metrics = {}
        # Train
        if step > training_config.learning_starts:
            for _ in range(training_config.gradient_steps):
                train_metrics = agent.train_step()
                
        # Episode end
        if done:
            # Log episode metrics
            logger.log({
                'episode/reward': episode_reward,
                'episode/length': episode_length,
                'episode/count': agent.episode_count,
                'exploration_noise': agent.exploration_noise,
                **info
            }, step)
            
            # Reset
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            agent.episode_count += 1
            
            # Update exploration
            agent.update_exploration()
        else:
            obs = next_obs
            
        # Evaluation
        if step % training_config.eval_frequency == 0 and train_metrics:
            eval_metrics = evaluate_agent(
                agent, eval_env, 
                num_episodes=training_config.num_eval_episodes
            )
            logger.log(eval_metrics, step)
            
        # Save checkpoint
        if step % training_config.save_frequency == 0:
            save_checkpoint(
                agent,
                f"{env_name}_step{step}.pt",
                additional_info={'step': step}
            )
            
        # Log training metrics
        if step % training_config.log_frequency == 0 and len(train_metrics) > 0:
            logger.log(train_metrics, step)
            
    # Final evaluation and video
    final_eval = evaluate_agent(agent, eval_env, num_episodes=20)
    logger.log(final_eval, total_timesteps)
    
    create_video(
        agent, eval_env,
        f"{env_name}_final.mp4",
        num_episodes=3
    )
    
    # Save final model
    save_checkpoint(agent, f"{env_name}_final.pt")
    
    # Create plots
    plot_training_curves(
        logger.log_file,
        save_path=f"plots/{env_name}_training.png",
        metrics=['episode/reward', 'elbo', 'policy_loss', 'value_loss']
    )
    
    logger.finish()
    
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                       choices=['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 
                               'Ant-v4', 'Humanoid-v4', 'HumanoidStandup-v4'])
    parser.add_argument('--pixels', action='store_true', 
                       help='Use pixel observations')
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_diffusion_active_inference(
        env_name=args.env,
        use_pixels=args.pixels,
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device
    )