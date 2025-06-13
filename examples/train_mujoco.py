"""
Training script for Diffusion Active Inference on MuJoCo with GPU-Optimized Parallel Data Collection
Uses GPUCentralizedCollector for faster diffusion inference during collection
"""

import torch
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Any
import time

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
# Use GPU-optimized collector instead of regular parallel collector
from active_inference_diffusion.utils.async_collector import GPUCentralizedCollector
from active_inference_diffusion.utils.util import visualize_reconstruction
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
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


def create_env_fn(env_name: str, use_pixels: bool, base_seed: int):
    """Create environment factory function for GPU collector"""
    def make_env(env_seed: int):
        def _init():
            if use_pixels:
                env = make_pixel_mujoco(
                    env_name,
                    width=84,
                    height=84,
                    frame_stack=3,
                    action_repeat=2,
                    seed=base_seed + env_seed,
                    normalize=True
                )
            else:
                env = gym.make(env_name, render_mode='rgb_array')
                env = NormalizeObservation(env)
                env = ActionRepeat(env, repeat=2)
                env.reset(seed=base_seed + env_seed)
                env.action_space.seed(base_seed + env_seed)
                env.observation_space.seed(base_seed + env_seed)
            return env
        return _init
    return make_env


def get_gpu_utilization(device: int = 0) -> float:
    """
    Get GPU utilization percentage using torch.cuda methods
    
    Fix C7: torch.cuda.utilization() doesn't exist, use memory utilization as proxy
    """
    if torch.cuda.is_available():
        # Get memory stats
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Calculate utilization as percentage of allocated memory
        # This is a proxy for GPU utilization
        utilization = (allocated_memory / total_memory) * 100.0
        
        return utilization
    return 0.0


def train_diffusion_active_inference(
    env_name: str = "HalfCheetah-v4",
    use_pixels: bool = False,
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_parallel_envs: int = 8,  # Increased from 6 to leverage GPU better
    use_gpu_collector: bool = True  # Flag to use GPU-optimized collector
):
    """Main training function with GPU-optimized parallel data collection"""
    
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Ensure we're using the right device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        use_gpu_collector = False  # Can't use GPU collector without CUDA
    
    # Create single environment for initialization and evaluation
    env = setup_environment(env_name, use_pixels, seed)
    eval_env = setup_environment(env_name, use_pixels, seed + 100)
    
    # Create configurations
    config = ActiveInferenceConfig(
        env_name=env_name,
        latent_dim=50,
        hidden_dim=256,
        learning_rate=5e-5,
        batch_size=2048,
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
        num_diffusion_steps=25,  # Can be reduced to 10 for GPU collector
        beta_schedule="cosine",
        beta_start=1e-4,
        beta_end=0.02
    )
    
    buffer_size = 100_000 if not use_pixels else 50_000
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
        num_parallel_envs=num_parallel_envs
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
    if hasattr(agent, 'encoder'):
        agent.encoder = agent.encoder.to(device)    
    # Create data collector
    if use_gpu_collector and device == "cuda":
        print(f"Creating GPU-optimized collector with {num_parallel_envs} environments...")
        print("Diffusion inference will run on GPU while environments step on CPU")
        
        # Create environment factory function
        env_fn = create_env_fn(env_name, use_pixels, seed)
        
        # Create GPU-optimized collector
        collector = GPUCentralizedCollector(
            env_fn=env_fn(0),  # Base environment function
            agent=agent,
            num_envs=num_parallel_envs,
            max_queue_size=32,
            use_mixed_precision=False  # Disable mixed precision for stability
        )
        
        # Override diffusion steps for faster collection
        collector.gpu_inference.max_diffusion_steps = 30  # Reduced from 40
        
    else:
        # Fallback to CPU-based parallel collector
        print(f"Creating CPU parallel collector with {num_parallel_envs} environments...")
        from active_inference_diffusion.envs.parallel_wrapper import create_parallel_collector
        
        collector = create_parallel_collector(
            env_name=env_name,
            agent=agent,
            num_envs=num_parallel_envs,
            use_pixels=use_pixels,
            seed=seed
        )
    
    # Create logger
    collector_type = "gpu" if use_gpu_collector and device == "cuda" else "cpu"
    logger = Logger(
        use_wandb=True,
        project_name="diffusion-active-inference-mujoco",
        experiment_name=f"{env_name}_{'pixels' if use_pixels else 'states'}_{collector_type}_parallel{num_parallel_envs}_seed{seed}",
        config={
            **config.__dict__,
            **training_config.__dict__,
            'num_parallel_envs': num_parallel_envs,
            'collector_type': collector_type
        }
    )
    
    # Create plots directory
    Path("plots").mkdir(exist_ok=True)
    
    # Training metrics tracking
    steps_collected = 0
    episodes_completed = 0
    
    print(f"Starting training with {num_parallel_envs} parallel environments...")
    print(f"Using {collector_type.upper()} collector for data collection")
    if use_gpu_collector and device == "cuda":
        print(f"Diffusion steps are {collector.gpu_inference.max_diffusion_steps} for faster GPU inference")
    
    # Main training loop
    try:
        while steps_collected < total_timesteps:
            # Calculate collection steps
            collection_steps = min(
                training_config.train_frequency * config.batch_size,
                total_timesteps - steps_collected
            )
            
            # Track collection time
            collection_start = time.time()
            
            # Collect data using appropriate method
            if use_gpu_collector and device == "cuda":
                # GPU-optimized collection
                collection_stats = collector.collect_parallel_batch(
                    num_steps=collection_steps,
                    replay_buffer=agent.replay_buffer
                )
            else:
                # CPU parallel collection
                collection_stats = collector.collect_steps(
                    num_steps=collection_steps,
                    replay_buffer=agent.replay_buffer
                )
            
            collection_time = time.time() - collection_start
            collection_rate = collection_stats['steps_collected'] / collection_time
            
            steps_collected += collection_stats['steps_collected']
            episodes_completed += collection_stats['episodes_completed']
            agent.total_steps = steps_collected
            
            # Log collection statistics
            log_data = {
                'parallel/collection_rate': collection_rate,
                'parallel/collection_time': collection_time,
                'parallel/steps_per_env': collection_stats['steps_collected'] / num_parallel_envs,
                'episode/count': episodes_completed,
                'exploration_noise': agent.exploration_noise,
            }
            
            if 'mean_episode_reward' in collection_stats:
                log_data.update({
                    'episode/reward': collection_stats['mean_episode_reward'],
                    'episode/reward_std': collection_stats['std_episode_reward'],
                })
            
            # Log GPU utilization if using GPU collector
            if use_gpu_collector and device == "cuda":
                # Use our custom GPU utilization function
                gpu_util = get_gpu_utilization()
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                log_data.update({
                    'gpu/utilization': gpu_util,
                    'gpu/memory_gb': gpu_memory,
                })
                
                # Log inference statistics if available
                if hasattr(collector.gpu_inference, 'inference_times') and collector.gpu_inference.inference_times:
                    avg_inference_time = np.mean(collector.gpu_inference.inference_times[-100:])
                    log_data['gpu/avg_inference_time'] = avg_inference_time
            
            logger.log(log_data, steps_collected)
            
            # Training phase
            if steps_collected > training_config.learning_starts:
                training_start = time.time()
                
                # Perform gradient updates
                num_updates = int(training_config.gradient_steps * collection_stats['steps_collected'])
                
                train_metrics = {}
                for _ in range(num_updates):
                    metrics = agent.train_step()
                    for k, v in metrics.items():
                        if k not in train_metrics:
                            train_metrics[k] = []
                        train_metrics[k].append(v)
                
                # Average training metrics
                avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
                
                training_time = time.time() - training_start
                avg_train_metrics['training/time'] = training_time
                avg_train_metrics['training/updates_per_second'] = num_updates / training_time
                
                # Log training metrics
                if steps_collected % training_config.log_frequency < collection_steps:
                    logger.log(avg_train_metrics, steps_collected)
            
            # Update exploration noise
            agent.update_exploration()
            
            # Visualization of reconstruction
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
                    additional_info={
                        'step': steps_collected,
                        'collector_type': collector_type
                    }
                )
            
            # Print progress with collection rate
            if steps_collected % 10000 == 0:
                print(f"Progress: {steps_collected}/{total_timesteps} steps ({100*steps_collected/total_timesteps:.1f}%)")
                print(f"Collection rate: {collection_rate:.1f} steps/second")
                if use_gpu_collector and device == "cuda":
                    print(f"GPU utilization: {gpu_util:.1f}%, Memory: {gpu_memory:.1f}GB")
    
    finally:
        print("Cleaning up data collector...")
        if hasattr(collector, 'close'):
            collector.close()
        elif hasattr(collector, 'gpu_inference'):
            try:
                collector.gpu_inference.stop()
            except:
                pass
        
        # Explicitly close environments
        try:
            env.close()
            eval_env.close()
        except:
            pass
        
        # Force garbage collection
        import gc
        gc.collect()   

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
    save_checkpoint(agent, f"{env_name}_final.pt", additional_info={'collector_type': collector_type})
    
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
    parser.add_argument('--num_parallel_envs', type=int, default=4,
                        help='Number of parallel environments for data collection')
    parser.add_argument('--no_gpu_collector', action='store_true',
                        help='Disable GPU-optimized collector (use CPU parallel collector)')
    
    args = parser.parse_args()
    
    try:
        train_diffusion_active_inference(
            env_name=args.env,
            use_pixels=args.pixels,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
            num_parallel_envs=args.num_parallel_envs,
            use_gpu_collector=not args.no_gpu_collector  # Use GPU collector by default
        )
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()