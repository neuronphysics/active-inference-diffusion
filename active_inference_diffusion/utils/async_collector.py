"""
GPU-Optimized Parallel Data Collection Architecture
Separates environment stepping (CPU) from diffusion inference (GPU)
"""

import torch
import torch.multiprocessing as mp
from threading import Thread
from queue import Queue, Empty
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import time
import gymnasium as gym
from pathlib import Path
from active_inference_diffusion.agents.state_agent import DiffusionStateAgent
from active_inference_diffusion.agents.pixel_agent import DiffusionPixelAgent
from collections import deque
import torch.nn.functional as F
from contextlib import contextmanager

class GPUCentralizedCollector:
    """
    Hybrid CPU-GPU architecture optimizing computational resource allocation:
    - Environment stepping and rendering: CPU workers  
    - Diffusion inference and policy evaluation: GPU main process
    - Asynchronous pipeline with overlapped computation
    """
    
    def __init__(
        self,
        env_fn,
        agent,
        num_envs: int = 8,
        max_queue_size: int = 32,
        use_mixed_precision: bool = True
    ):
        self.num_envs = num_envs
        # Use agent's device instead of hardcoding 'cuda'
        self.device = agent.device if hasattr(agent, 'device') else torch.device('cuda')
        self.agent = agent

        if hasattr(agent, 'active_inference'):
            agent.active_inference = agent.active_inference.to(self.device)
            agent.active_inference.eval()
            
        if hasattr(agent, 'encoder'):
            agent.encoder = agent.encoder.to(self.device)
            agent.encoder.eval()
        # Get the actual configured diffusion steps from the agent
        actual_diffusion_steps = agent.config.diffusion.num_diffusion_steps
        
        # Initialize async GPU inference pipeline
        self.gpu_inference = AsyncGPUInference(
            agent=agent,
            max_queue_size=max_queue_size,
            use_mixed_precision=use_mixed_precision,
            max_diffusion_steps=actual_diffusion_steps
        )
        
        # Create environment workers (CPU-only)
        self.current_observations = [None] * num_envs
        self.env_workers = self._create_env_workers(env_fn)
        
        # State management
        
        self.env_states = ['ready'] * num_envs
        
        
    def _create_env_workers(self, env_fn) -> List['EnvironmentWorker']:
        """Create CPU-bound environment workers for stepping and rendering"""
        workers = []
        
        for i in range(self.num_envs):
            worker = EnvironmentWorker(
                env_fn=env_fn,
                worker_id=i,
                seed=42 + i
            )
            worker.start()
            workers.append(worker)
            
        # Initialize environments and get first observations
        for i, worker in enumerate(workers):
            obs, _ = worker.reset()
            self.current_observations[i] = obs
        self.observation_shape = self.current_observations[0].shape           
        return workers
    
    def collect_parallel_batch(
        self, 
        num_steps: int,
        replay_buffer
    ) -> Dict[str, float]:
        """
        High-throughput parallel collection with GPU-batched inference
        
        Pipeline Architecture:
        1. Batch observations → GPU inference → actions
        2. Distribute actions → environment stepping (parallel)
        3. Collect results → update buffer → repeat
        """
        
        steps_collected = 0
        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        completed_episodes = []
        
        # Start GPU inference pipeline
        self.gpu_inference.start()
        
        try:
            while steps_collected < num_steps:
                # Batch current observations for GPU inference
                obs_batch = self._prepare_observation_batch()
                
                # Submit to GPU inference (non-blocking)
                inference_future = self.gpu_inference.submit_batch(obs_batch)
                
                # Overlap: while GPU computes, prepare next batch or handle results
                actions_batch = inference_future.get(timeout=20.0)  # GPU inference result
                
                # Distribute actions to environment workers
                step_futures = self._distribute_actions(actions_batch)
                
                # Collect environment step results
                results = self._collect_step_results(step_futures)
                
                # Process results and update buffer
                for env_idx, (obs, action, reward, next_obs, done, info) in enumerate(results):
                    if obs is not None and action is not None:
                        replay_buffer.add(obs, action, reward, next_obs, done)
                        
                        episode_rewards[env_idx] += reward
                        episode_lengths[env_idx] += 1
                        steps_collected += 1
                        
                        if done:
                            # Episode completed
                            completed_episodes.append({
                                'reward': episode_rewards[env_idx],
                                'length': episode_lengths[env_idx]
                            })
                            episode_rewards[env_idx] = 0.0
                            episode_lengths[env_idx] = 0

                            # Properly handle reset and get new observation
                            reset_future = self.env_workers[env_idx].reset_async()
                            # Wait for reset to complete and get new observation
                            new_obs, reset_info = reset_future.result()
                            self.current_observations[env_idx] = new_obs
                        else:
                            self.current_observations[env_idx] = next_obs
                        
        finally:
            self.gpu_inference.stop()
            
        # Compute collection statistics
        stats = {
            'steps_collected': steps_collected,
            'episodes_completed': len(completed_episodes)
        }
        
        if completed_episodes:
            rewards = [ep['reward'] for ep in completed_episodes]
            stats.update({
                'mean_episode_reward': np.mean(rewards),
                'std_episode_reward': np.std(rewards),
                'mean_episode_length': np.mean([ep['length'] for ep in completed_episodes])
            })
            
        return stats
    
    def _prepare_observation_batch(self) -> torch.Tensor:
        """Vectorize observations for batched GPU inference"""
        obs_list = []
        for obs in self.current_observations:
            if obs is not None:
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.from_numpy(obs).float()
                else:
                    obs_tensor = obs
                obs_list.append(obs_tensor)
            else:
                # Handle None observations with zeros using stored shape
                shape = tuple(self.observation_shape) if isinstance(self.observation_shape, np.ndarray) else self.observation_shape
                obs_list.append(torch.zeros(shape))         
        return torch.stack(obs_list).to(self.device)
    
    def _distribute_actions(self, actions_batch: torch.Tensor) -> List:
        """Distribute actions to environment workers asynchronously"""
        actions_cpu = actions_batch.cpu().numpy()
        futures = []
        
        for env_idx, action in enumerate(actions_cpu):
            future = self.env_workers[env_idx].step_async(action)
            futures.append((env_idx, future))
            
        return futures
    
    def _collect_step_results(self, step_futures) -> List[Tuple]:
        """Collect results from environment workers"""
        results = []
        
        for env_idx, future_dict in step_futures:
            try:
                prev_obs = self.current_observations[env_idx]
                action = future_dict['action']
                step_result = future_dict['result'].result(timeout=8.0)

                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    # Fallback for older Gym versions
                    next_obs, reward, done, info = step_result
                
                results.append((prev_obs, action, reward, next_obs, done, info))
                
            except Exception as e:
                print(f"Environment {env_idx} step failed: {e}")
                results.append((None, None, 0.0, None, True, {}))
                
        return results
    
    def close(self):
        """Clean up with proper resource ordering"""
        print("Closing GPU collector...")
        # 1. First stop GPU inference
        if hasattr(self, 'gpu_inference'):
            self.gpu_inference.stop()
        
        # 2. Close environment workers
        for worker in self.env_workers:
            try:
                worker.close()
            except Exception as e:
                print(f"Error closing worker: {e}")
        
        # 3. Explicitly clear references
        self.env_workers = []
        self.current_observations = []

class AsyncGPUInference:
    """
    Dedicated GPU inference pipeline with stream parallelism and mixed precision
    Maximizes GPU utilization through continuous batched processing
    """
    
    def __init__(
        self, 
        agent,
        max_queue_size: int = 32,
        use_mixed_precision: bool = True,
        max_diffusion_steps: Optional[int] = None
    ):
        self.agent = agent
        self.max_queue_size = max_queue_size
        self.use_mixed_precision = use_mixed_precision
        self.max_diffusion_steps = max_diffusion_steps
        if max_diffusion_steps is None:
            self.max_diffusion_steps = agent.config.diffusion.num_diffusion_steps
        else:
            # Ensure we don't exceed the configured maximum
            configured_steps = agent.config.diffusion.num_diffusion_steps
            self.max_diffusion_steps = min(max_diffusion_steps, configured_steps)
            if max_diffusion_steps > configured_steps:
                print(f"Warning: Requested {max_diffusion_steps} diffusion steps, "
                      f"but model only supports {configured_steps}. Using {self.max_diffusion_steps}.")
        
        # Get action dimension from agent's action space
        self.action_dim = agent.action_space.shape[0]
        
        # Threading infrastructure
        self.inference_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue(maxsize=max_queue_size)
        self.shutdown_event = mp.Event()
        
        # CUDA optimization
        self.inference_stream = torch.cuda.Stream()
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Performance monitoring
        self.inference_times = []
        self.batch_sizes = []
        self.done =True
        
    def start(self):
        """Initialize GPU inference worker thread"""
        self.gpu_thread = Thread(target=self._gpu_inference_worker, daemon=True)
        self.gpu_thread.start()
        
    def stop(self):
        """Gracefully shutdown inference pipeline"""
        self.shutdown_event.set()
        self.inference_queue.put(None)  # Sentinel
        if hasattr(self, 'gpu_thread'):
            self.gpu_thread.join(timeout=10.0)
    
    def submit_batch(self, observations_batch: torch.Tensor) -> 'InferenceFuture':
        """Submit observation batch for GPU inference"""
        future = InferenceFuture()
        
        try:
            self.inference_queue.put((observations_batch, future), timeout=1.0)
        except:
            # Queue full - return dummy actions
            dummy_actions = torch.zeros(observations_batch.shape[0], self.action_dim)
            future.set_result(dummy_actions)
            
        return future
    
    def _gpu_inference_worker(self):
        """Dedicated GPU worker for continuous inference processing"""
        torch.cuda.set_device(0)  # Ensure GPU context
        
        with torch.cuda.stream(self.inference_stream):
            while not self.shutdown_event.is_set():
                try:
                    batch_item = self.inference_queue.get(timeout=0.1)
                    if batch_item is None:  # Shutdown sentinel
                        break
                        
                    observations_batch, future = batch_item
                    start_time = time.time()
                    
                    # Execute batched diffusion inference on GPU
                    actions_batch = self._batched_diffusion_inference(observations_batch)
                    
                    # Performance tracking
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.batch_sizes.append(observations_batch.shape[0])
                    
                    future.set_result(actions_batch)
                    
                except Empty:
                    continue
                except Exception as e:
                    print(f"GPU inference error: {e}")
                    if 'future' in locals():
                        future.set_error(e)
    
    def _batched_diffusion_inference(self, observations_batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorized diffusion inference across entire observation batch
        Core optimization: Single GPU call for all environments
        """
        batch_size = observations_batch.shape[0]
        
        with torch.no_grad():
            # Mixed precision context for memory efficiency
            if self.use_mixed_precision:
                with torch.amp.autocast(
                    device_type='cuda',
                    dtype=torch.float16):  # Use fp16 for mixed precision
                    actions_batch = self._inference_impl(observations_batch, batch_size)
            else:
                actions_batch = self._inference_impl(observations_batch, batch_size)
                
        return actions_batch.float()  # Ensure fp32 output
    
    def _inference_impl(self, observations_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Implementation of inference logic"""
        # Encode observations (if pixel-based)
        if hasattr(self.agent, 'encoder'):
            encoded_obs = self.agent.encoder(observations_batch)
        else:
            encoded_obs = observations_batch
        
        # Batched belief generation via reverse diffusion
        latents_batch = self._batch_diffusion_sampling(
            encoded_obs, 
            num_steps=self.max_diffusion_steps
        )
        
        # Batched policy evaluation
        actions_batch, _, _ = self.agent.active_inference.policy_network(
            latents_batch, 
            deterministic=False
        )
        
        return actions_batch
    
    def _batch_diffusion_sampling(
        self, 
        observations_batch: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Optimized batched reverse diffusion with reduced steps
        Implements parallel sampling across batch dimension
        """
        batch_size = observations_batch.shape[0]
        latent_dim = self.agent.config.latent_dim
        if num_steps is None:
            num_steps = self.max_diffusion_steps
        else:
            # Ensure we don't exceed maximum configured steps
            num_steps = min(num_steps, self.max_diffusion_steps)
        if num_steps <= 0:
            raise ValueError(f"Invalid num_steps: {num_steps}. Must be positive")
       
        # Initialize noise for entire batch
        z_batch = torch.randn(batch_size, latent_dim, device=observations_batch.device)
        
        max_index = self.agent.config.diffusion.num_diffusion_steps - 1
        
        for step in reversed(range(num_steps)):
            t_batch = torch.full((batch_size,), step, device=observations_batch.device, dtype=torch.float32)
            
            # Clamp float values first
            t_batch = torch.clamp(t_batch, min=0, max=max_index)
            
            # Convert to long and clamp again to ensure integer bounds
            t_batch_long = t_batch.long()
            t_batch_long = torch.clamp(t_batch_long, 0, max_index)
            
            # Additional boundary check
            if (t_batch_long < 0).any() or (t_batch_long > max_index).any():
                invalid_min = t_batch_long.min().item()
                invalid_max = t_batch_long.max().item()
                raise RuntimeError(
                    f"Time steps out of bounds after clamping: "
                    f"min={invalid_min}, max={invalid_max}, "
                    f"allowed=[0, {max_index}]"
                )
            
            # For continuous time models
            if hasattr(self.agent.active_inference.latent_diffusion, 'continuous_time') and \
               self.agent.active_inference.latent_diffusion.continuous_time:
                t_continuous = t_batch.float() / max_index
                score_batch = self.agent.active_inference.latent_score_network(
                    z_batch, t_continuous, observations_batch
                )
            else:
                # Use clamped long indices for discrete time
                score_batch = self.agent.active_inference.latent_score_network(
                    z_batch, t_batch_long.float(), observations_batch
                )
            
            # Use clamped long indices for diffusion update
            z_batch = self.agent.active_inference.latent_diffusion.p_sample(
                z_batch, t_batch_long, score_batch, deterministic=False
            )
        
        return z_batch


class InferenceFuture:
    """Thread-safe future for asynchronous inference results"""
    
    def __init__(self):
        self.result = None
        self.error = None
        self.ready = mp.Event()
    
    def set_result(self, result):
        self.result = result
        self.ready.set()
    
    def set_error(self, error):
        self.error = error
        self.ready.set()
    
    def get(self, timeout=None):
        if self.ready.wait(timeout):
            if self.error:
                raise self.error
            return self.result
        else:
            raise TimeoutError("Inference timeout")


class EnvironmentWorker:
    """CPU-bound environment worker for stepping and rendering only"""
    
    def __init__(self, env_fn, worker_id: int, seed: int):
        self.env_fn = env_fn
        self.worker_id = worker_id
        self.seed = seed
        self.executor = None
        self.env = None
        
    def start(self):
        """Initialize worker with dedicated thread pool"""
        from concurrent.futures import ThreadPoolExecutor
        self.executor = ThreadPoolExecutor(max_workers=2,  # Increased from 1 for better parallelism
                                         thread_name_prefix=f'env-{self.worker_id}')
        self.env = self.env_fn()
        self.env.reset(seed=self.seed)
    
    def reset(self):
        """Synchronous environment reset"""
        return self.env.reset()
    
    def reset_async(self):
        """Asynchronous environment reset"""
        # Return proper future that can be used to get result
        return self.executor.submit(self.env.reset)
    
    def step_async(self, action):
        """Asynchronous environment step"""
        def step_with_action():
            return self.env.step(action)
        
        future = self.executor.submit(step_with_action)
        return {'action': action, 'result': future}
    

def create_gpu_collector(
    env_name: str,
    agent: Union['DiffusionStateAgent', 'DiffusionPixelAgent'],
    num_envs: int = 8,
    use_pixels: bool = False,
    seed: int = 0,
    max_diffusion_steps: Optional[int] = None
) -> GPUCentralizedCollector:
    """
    Create GPU-optimized collector with proper environment setup
    
    Args:
        env_name: Name of the MuJoCo environment
        agent: The active inference agent
        num_envs: Number of parallel environments
        use_pixels: Whether to use pixel observations
        seed: Random seed
        max_diffusion_steps: Override default diffusion steps for speed
        
    Returns:
        Configured GPUCentralizedCollector
    """
    from active_inference_diffusion.envs.wrappers import NormalizeObservation, ActionRepeat
    from active_inference_diffusion.envs.pixel_wrappers import make_pixel_mujoco
    
    def make_env(env_seed: int):
        """Create properly configured environment"""
        def _init():
            if use_pixels:
                env = make_pixel_mujoco(
                    env_name,
                    width=84,
                    height=84,
                    frame_stack=3,
                    action_repeat=2,
                    seed=seed + env_seed,
                    normalize=True
                )
            else:
                env = gym.make(env_name, render_mode='rgb_array')
                env = NormalizeObservation(env)
                env = ActionRepeat(env, repeat=2)
                env.reset(seed=seed + env_seed)
                env.action_space.seed(seed + env_seed)
                env.observation_space.seed(seed + env_seed)
            return env
        return _init
    
    # Create GPU-centralized collector
    collector = GPUCentralizedCollector(
        env_fn=make_env(0),
        agent=agent,
        num_envs=num_envs,
        use_mixed_precision=False  # Use full precision for stability
    )
    
    # Override diffusion steps if specified
    if max_diffusion_steps is not None:
        collector.gpu_inference.max_diffusion_steps = max_diffusion_steps
    
    return collector


class CollectorWrapper:
    """
    Unified interface for both GPU and CPU collectors
    Allows seamless switching between collection strategies
    """
    
    def __init__(self, collector, collector_type: str = "gpu"):
        self.collector = collector
        self.collector_type = collector_type
        
    def collect_steps(self, num_steps: int, replay_buffer) -> Dict[str, float]:
        """Unified collection interface"""
        if self.collector_type == "gpu":
            return self.collector.collect_parallel_batch(num_steps, replay_buffer)
        else:
            return self.collector.collect_steps(num_steps, replay_buffer)
    
    def close(self):
        """Clean up resources with proper context management"""
        try:
            if self.env is not None:
                # Explicitly render once to ensure context is active
                try:
                    self.env.render()
                except:
                    pass
                # Close environment
                self.env.close()
                self.env = None
        except Exception as e:
            print(f"Error closing environment: {e}")
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=False)
                self.executor = None

    def get_stats(self) -> Dict[str, float]:
        """Get collection statistics"""
        stats = {}
        
        if self.collector_type == "gpu" and hasattr(self.collector.gpu_inference, 'inference_times'):
            if self.collector.gpu_inference.inference_times:
                stats['avg_inference_time'] = np.mean(self.collector.gpu_inference.inference_times[-100:])
                stats['total_inferences'] = len(self.collector.gpu_inference.inference_times)
                
        return stats