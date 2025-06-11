"""
Training script for Diffusion Active Inference on MuJoCo with Parallel Data Collection
Fixed version with proper CUDA multiprocessing support
"""

import torch
import torch.multiprocessing as mp
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
from active_inference_diffusion.envs.parallel_wrapper import create_parallel_collector
from active_inference_diffusion.utils.util import visualize_reconstruction
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Ensure CUDA errors are raised immediately
os.environ['MUJOCO_GL'] = 'egl' 
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
        print("Observation space for pixel:", env.observation_space.shape)
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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_parallel_envs: int = 6  # Use 6 parallel environments as in config
):
    """Main training function with parallel data collection"""
    
    # Set multiprocessing start method for CUDA compatibility
    # This must be done before any CUDA operations
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set, which is fine
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Ensure we're using the right device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Create single environment for initialization and evaluation
    env = setup_environment(env_name, use_pixels, seed)
    eval_env = setup_environment(env_name, use_pixels, seed + 100)
    
    # Create configurations
    config = ActiveInferenceConfig(
        env_name=env_name,
        latent_dim=50,
        hidden_dim=256,
        learning_rate=5e-5,
        batch_size=512,
        efe_horizon=5,
        epistemic_weight=0.1,
        pragmatic_weight=1.0,
        consistency_weight=0.1,
        kl_weight=0.5,
        diffusion_weight=1.0,
        pixel_observation=use_pixels,
        device=device
    )
    
    # Enhanced diffusion config
    config.diffusion = DiffusionConfig(
        num_diffusion_steps=40,  # Fewer steps for faster inference
        beta_schedule="cosine",
        beta_start=1e-4,
        beta_end=0.02
    )
    buffer_size = 100_000 if not use_pixels else 50_000  # Smaller buffer for pixel data 
    training_config = TrainingConfig(
        total_timesteps=total_timesteps,
        eval_frequency=10_000,
        save_frequency=50_000,
        log_frequency=1_000,
        buffer_size=buffer_size,
        learning_starts=5_000,
        gradient_steps=4,
        exploration_noise=0.1,
        exploration_decay=0.999,
        num_parallel_envs=num_parallel_envs  # Set number of parallel envs
    )
    
    # Create agent
    print(f"Creating agent on device: {device}")
    if use_pixels:
        pixel_config = PixelObservationConfig()
        agent = DiffusionPixelAgent(
            env, config, training_config, pixel_config
        )
    else:
        agent = DiffusionStateAgent(env, config, training_config)
    
    # Ensure agent is on the correct device
    agent.device = torch.device(device)
    agent.active_inference = agent.active_inference.to(device)
    
    # Create parallel data collector
    print(f"Creating parallel data collector with {num_parallel_envs} environments...")
    print("Note: Worker processes will use CPU for action selection while main process uses GPU for training")
    
    collector = create_parallel_collector(
        env_name=env_name,
        agent=agent,
        num_envs=num_parallel_envs,
        use_pixels=use_pixels,
        seed=seed
    )
    
    # Create logger
    logger = Logger(
        use_wandb=True,
        project_name="diffusion-active-inference-mujoco",
        experiment_name=f"{env_name}_{'pixels' if use_pixels else 'states'}_parallel{num_parallel_envs}_seed{seed}",
        config={
            **config.__dict__,
            **training_config.__dict__,
            'num_parallel_envs': num_parallel_envs
        }
    )
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Training metrics tracking
    steps_collected = 0
    episodes_completed = 0
    
    print(f"Starting training with {num_parallel_envs} parallel environments...")
    print(f"Main training process using device: {device}")
    
    # Main training loop
    try:
        while steps_collected < total_timesteps:
            # Calculate how many steps to collect this iteration
            # Collect enough steps to do multiple gradient updates
            collection_steps = training_config.train_frequency * config.batch_size
            
            # Don't exceed total timesteps
            collection_steps = min(collection_steps, total_timesteps - steps_collected)
            
            # Collect data in parallel across multiple environments
            collection_stats = collector.collect_steps(
                num_steps=collection_steps,
                replay_buffer=agent.replay_buffer
            )
            
            steps_collected += collection_stats['steps_collected']
            episodes_completed += collection_stats['episodes_completed']
            agent.total_steps = steps_collected
            
            # Log collection statistics
            if 'mean_episode_reward' in collection_stats:
                logger.log({
                    'episode/reward': collection_stats['mean_episode_reward'],
                    'episode/reward_std': collection_stats['std_episode_reward'],
                    'episode/count': episodes_completed,
                    'exploration_noise': agent.exploration_noise,
                    'parallel/collection_rate': collection_stats['steps_collected'] / num_parallel_envs,
                }, steps_collected)
            
            # Training phase - perform gradient updates if we have enough data
            if steps_collected > training_config.learning_starts:
                # Perform multiple gradient steps per collection phase
                num_updates = int(training_config.gradient_steps * collection_stats['steps_collected'])
                
                train_metrics = {}
                for _ in range(num_updates):
                    metrics = agent.train_step()
                    # Aggregate metrics
                    for k, v in metrics.items():
                        if k not in train_metrics:
                            train_metrics[k] = []
                        train_metrics[k].append(v)
                
                # Average the training metrics
                avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
                
                # Log training metrics
                if steps_collected % training_config.log_frequency < collection_steps:
                    logger.log(avg_train_metrics, steps_collected)
            
            # Update exploration noise
            agent.update_exploration()
            
            # Visualization of reconstruction (for debugging)
            if steps_collected > training_config.learning_starts and \
               steps_collected % 5000 < collection_steps and len(agent.replay_buffer) > 0:
                sample_batch = agent.replay_buffer.sample(min(4, len(agent.replay_buffer)))
                sample_obs = sample_batch['observations']
                recon_error = visualize_reconstruction(
                    agent,
                    sample_obs,
                    f"plots/reconstruction_step_{steps_collected}.png"
                )
                logger.log({'reconstruction_error': recon_error}, steps_collected)
            
            # Evaluation
            if steps_collected % training_config.eval_frequency < collection_steps:
                print(f"Evaluating at step {steps_collected}...")
                eval_metrics = evaluate_agent(
                    agent, eval_env,
                    num_episodes=training_config.num_eval_episodes
                )
                logger.log(eval_metrics, steps_collected)
                print(f"Eval reward: {eval_metrics['eval/mean_reward']:.2f} ± {eval_metrics['eval/std_reward']:.2f}")
            
            # Save checkpoint
            if steps_collected % training_config.save_frequency < collection_steps:
                save_checkpoint(
                    agent,
                    f"{env_name}_step{steps_collected}.pt",
                    additional_info={'step': steps_collected}
                )
                
            # Print progress
            if steps_collected % 10000 == 0:
                print(f"Progress: {steps_collected}/{total_timesteps} steps ({100*steps_collected/total_timesteps:.1f}%)")
    
    finally:
        # Always cleanup parallel collector
        print("Cleaning up parallel data collector...")
        collector.close()
    
    # Final evaluation and video
    print("Running final evaluation...")
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
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                       choices=['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 
                               'Ant-v4', 'Humanoid-v4', 'HumanoidStandup-v4'])
        parser.add_argument('--pixels', action='store_true', 
                            help='Use pixel observations')
        parser.add_argument('--timesteps', type=int, default=1_000_000)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--num_parallel_envs', type=int, default=2,
                            help='Number of parallel environments for data collection')
    
        args = parser.parse_args()
    
        train_diffusion_active_inference(
                                        env_name=args.env,
                                        use_pixels=args.pixels,
                                        total_timesteps=args.timesteps,
                                        seed=args.seed,
                                        device=args.device,
                                        num_parallel_envs=args.num_parallel_envs
                                        )
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()