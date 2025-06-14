"""
GPU-Optimized Parallel Data Collection Architecture
Separates environment stepping (CPU) from diffusion inference (GPU)
"""

import torch
import torch.multiprocessing as mp
from threading import Thread, Event
from queue import Queue, Empty, Full
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import time
import gymnasium as gym
from pathlib import Path
from active_inference_diffusion.agents.state_agent import DiffusionStateAgent
from active_inference_diffusion.agents.pixel_agent import DiffusionPixelAgent
from collections import deque
import traceback
from active_inference_diffusion.envs.vec_env import ShmemVectorEnv, SubprocVectorEnv
"""
GPU-Optimized Parallel Data Collection using SubprocVectorEnv
"""


class GPUCentralizedCollector:
    """
    Hybrid CPU-GPU architecture using robust vectorized environments
    
    This implementation leverages the battle-tested SubprocVectorEnv for environment
    management while maintaining your sophisticated GPU inference pipeline.
    """
    
    def __init__(
        self,
        env_fn: Callable[[], Any],
        agent: Optional[Any] = None,
        num_envs: int = 8,
        max_queue_size: int = 32,
        use_mixed_precision: bool = True,
        use_shared_memory: bool = True  # Enable for pixel observations
    ):
        self.num_envs = num_envs
        self.device = agent.device if hasattr(agent, 'device') else torch.device('cuda')
        self.agent = agent
        self.use_shared_memory = use_shared_memory
        self._closing = False  # Add flag to track if we're closing
        self._pending_futures = []  # Track pending inference futures
        
        # Ensure agent components are on GPU and in eval mode
        if hasattr(agent, 'active_inference'):
            agent.active_inference = agent.active_inference.to(self.device)
            agent.active_inference.eval()
            
        if hasattr(agent, 'encoder'):
            agent.encoder = agent.encoder.to(self.device)
            agent.encoder.eval()
        
        # Create environment functions with proper seeding
        env_fns = [self._make_env_fn(env_fn, i) for i in range(num_envs)]
        
        # Choose the appropriate vectorized environment class
        if use_shared_memory:
            # ShmemVectorEnv is optimal for pixel observations
            # It avoids copying large arrays between processes
            self.vec_env = ShmemVectorEnv(env_fns)
            print(f"Using ShmemVectorEnv for efficient pixel observation transfer")
        else:
            # SubprocVectorEnv is fine for low-dimensional state observations
            self.vec_env = SubprocVectorEnv(env_fns, context='spawn')
            print(f"Using SubprocVectorEnv with spawn context")
        
        # Initialize with first observations
        self.current_observations, self.current_infos = self.vec_env.reset()
        
        # Store observation and action shapes for validation
        self.observation_shape = self.current_observations[0].shape
        self.action_shape = self.vec_env.action_space.shape
        
        # Initialize GPU inference pipeline
        actual_diffusion_steps = agent.config.diffusion.num_diffusion_steps
        self.gpu_inference = AsyncGPUInference(
            agent=agent,
            max_queue_size=max_queue_size,
            use_mixed_precision=use_mixed_precision,
            max_diffusion_steps=actual_diffusion_steps
        )
        
        # Performance monitoring
        self.step_times = []
        self.inference_times = []
        self.consecutive_timeouts = 0
        self.max_consecutive_timeouts = 3
   
    def _make_env_fn(self, base_env_fn: Callable, worker_id: int, seed: int = 42) -> Callable:
        """
        Create environment function with proper seeding and initialization
        
        This ensures each subprocess gets a properly configured environment
        with a unique seed to avoid correlation between parallel environments.
        """
        def _init():
            env = base_env_fn()
            
            # Set unique seed for this environment
            env_seed = seed + worker_id * 1000  # Large offset to avoid overlap
            try:
                # Try the new API first
                env.reset(seed=env_seed)
            except TypeError:
                if hasattr(env, 'seed'):
                    env.seed(env_seed)
                env.reset()
            
            # Seed action and observation spaces if possible
            if hasattr(env.action_space, 'seed'):
                env.action_space.seed(env_seed + 1)
            if hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(env_seed + 2)
            
            return env
        return _init
    
    def collect_parallel_batch(
        self, 
        num_steps: int,
        replay_buffer
    ) -> Dict[str, float]:
        """Main collection loop with improved error handling"""
        steps_collected = 0
        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        completed_episodes = []
        
        # Start GPU inference pipeline
        self.gpu_inference.start()
        
        try:
            while steps_collected < num_steps and not self._closing:
                loop_start = time.time()
                
                # === GPU PHASE: Batched Inference ===
                obs_batch = self._prepare_observation_batch(self.current_observations)
                
                # Submit to GPU for diffusion + policy inference
                inference_future = self.gpu_inference.submit_batch(obs_batch)
                self._pending_futures.append(inference_future)  # Track pending future
                
                # Wait for GPU to complete with timeout handling
                inference_start = time.time()
                try:
                    # Use dynamic timeout based on recent performance
                    dynamic_timeout = max(60.0, np.mean(self.inference_times[-10:]) * 20) if self.inference_times else 60.0
                    
                    # Check if we're closing before waiting
                    if self._closing:
                        inference_future.cancel()
                        break
                        
                    actions_batch = inference_future.get(timeout=dynamic_timeout)
                    
                    # Remove from pending futures on success
                    if inference_future in self._pending_futures:
                        self._pending_futures.remove(inference_future)
                    
                    # Reset consecutive timeout counter on success
                    self.consecutive_timeouts = 0
                    
                except TimeoutError:
                    # Remove from pending futures
                    if inference_future in self._pending_futures:
                        self._pending_futures.remove(inference_future)
                        
                    self.consecutive_timeouts += 1
                    print(f"Inference timeout #{self.consecutive_timeouts} (timeout: {dynamic_timeout:.1f}s)")
                    
                    if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                        raise RuntimeError(f"Too many consecutive timeouts ({self.consecutive_timeouts})")
                    
                    # Use random actions as fallback
                    print("Using random actions as fallback...")
                    actions_batch = torch.tensor(
                        np.array([self.vec_env.action_space.sample() for _ in range(self.num_envs)]),
                        device=self.device
                    )
                
                inference_time = time.time() - inference_start
                self.inference_times.append(inference_time)
                
                # === CPU PHASE: Parallel Environment Stepping ===
                actions_np = actions_batch.cpu().numpy()
                
                # Validate action shape
                if actions_np.shape[1:] != self.action_shape:
                    print(f"Warning: action shape {actions_np.shape[1:]} doesn't match expected {self.action_shape}")
                    # Reshape or pad as needed
                    if actions_np.shape[-1] < self.action_shape[0]:
                        # Pad with zeros
                        pad_width = [(0, 0)] + [(0, self.action_shape[i] - actions_np.shape[i+1]) for i in range(len(self.action_shape))]
                        actions_np = np.pad(actions_np, pad_width, mode='constant')
                    else:
                        # Truncate
                        actions_np = actions_np[:, :self.action_shape[0]]
                
                # Step all environments in parallel
                step_start = time.time()
                step_results = self.vec_env.step(actions_np)
                next_observations, rewards, terminateds, truncateds, infos = step_results
                step_time = time.time() - step_start
                self.step_times.append(step_time)
                
                # === Data Processing Phase ===
                for i in range(self.num_envs):
                    replay_buffer.add(
                        self.current_observations[i],
                        actions_np[i],
                        rewards[i],
                        next_observations[i],
                        terminateds[i] or truncateds[i]
                    )
                    
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    steps_collected += 1
                    
                    if terminateds[i] or truncateds[i]:
                        completed_episodes.append({
                            'reward': episode_rewards[i],
                            'length': episode_lengths[i],
                            'env_id': infos[i].get('env_id', i)
                        })
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                
                # Update observations for next iteration
                self.current_observations = next_observations
                
                # Log performance every 100 steps
                if steps_collected % 100 == 0:
                    self._log_performance()
                
        except Exception as e:
            print(f"Error in collection loop: {e}")
            traceback.print_exc()
            raise
        finally:
            self.gpu_inference.stop()
        
        # Compute final statistics
        stats = self._compute_statistics(steps_collected, completed_episodes)
        return stats
        
    def _prepare_observation_batch(self, observations: np.ndarray) -> torch.Tensor:
        """
        Convert numpy observations to GPU tensor with proper handling
        
        The rltoolkit vectorized env already provides a batched array,
        so we just need to convert it to torch and move to GPU.
        """
        # Handle different observation types
        if isinstance(observations, np.ndarray):
            # Standard case: numpy array from vectorized env
            obs_tensor = torch.from_numpy(observations).float()
        elif isinstance(observations, list):
            # Edge case: list of observations (shouldn't happen with rltoolkit)
            obs_tensor = torch.stack([torch.from_numpy(o).float() for o in observations])
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")
        
        # Move to GPU
        return obs_tensor.to(self.device)
    
    def _log_performance(self):
        """Log performance metrics for monitoring"""
        if self.inference_times and self.step_times:
            avg_inference = np.mean(self.inference_times[-10:])
            avg_step = np.mean(self.step_times[-10:])
            total_time = avg_inference + avg_step
            
            print(f"Performance - Inference: {avg_inference:.3f}s, "
                  f"Step: {avg_step:.3f}s, "
                  f"FPS: {self.num_envs / total_time:.1f}")
    
    def _compute_statistics(self, steps_collected: int, 
                           completed_episodes: List[Dict]) -> Dict[str, float]:
        """Compute collection statistics"""
        stats = {
            'steps_collected': steps_collected,
            'episodes_completed': len(completed_episodes)
        }
        
        if completed_episodes:
            rewards = [ep['reward'] for ep in completed_episodes]
            lengths = [ep['length'] for ep in completed_episodes]
            stats.update({
                'mean_episode_reward': np.mean(rewards),
                'std_episode_reward': np.std(rewards),
                'mean_episode_length': np.mean(lengths),
                'min_episode_reward': np.min(rewards),
                'max_episode_reward': np.max(rewards),
            })
        
        if self.inference_times:
            stats['avg_inference_time'] = np.mean(self.inference_times)
            stats['inference_fps'] = self.num_envs / stats['avg_inference_time']
        
        return stats
    
    def close(self):
        """Clean up resources"""
        print("Closing GPU collector...")
        
        # Set closing flag to stop collection loop
        self._closing = True
        
        # Cancel all pending futures
        for future in self._pending_futures:
            try:
                future.cancel()
            except:
                pass
        self._pending_futures.clear()
        
        # Stop GPU inference first
        if hasattr(self, 'gpu_inference'):
            self.gpu_inference.stop()
        
        # Close vectorized environments
        if hasattr(self, 'vec_env'):
            self.vec_env.close()
        
        print("GPU collector closed successfully")

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
        self.shutdown_event = Event()
        
        # CUDA optimization
        self.inference_stream = torch.cuda.Stream()
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Performance monitoring
        self.inference_times = []
        self.batch_sizes = []
        self.error_count = 0
        self.max_errors = 10
        
        # Thread reference
        self.gpu_thread = None
        
    def start(self):
        """Initialize GPU inference worker thread"""
        if self.gpu_thread is not None and self.gpu_thread.is_alive():
            print("GPU inference thread already running")
            return
            
        self.shutdown_event.clear()
        self.gpu_thread = Thread(target=self._gpu_inference_worker, daemon=True)
        self.gpu_thread.start()
        
    def stop(self):
        """Gracefully shutdown inference pipeline"""
        if self.gpu_thread is None:
            return
            
        self.shutdown_event.set()
        
        # Send sentinel to wake up thread if it's waiting
        try:
            self.inference_queue.put(None, timeout=1.0)
        except Full:
            pass
            
        if self.gpu_thread.is_alive():
            self.gpu_thread.join(timeout=5.0)
            if self.gpu_thread.is_alive():
                print("Warning: GPU inference thread did not shut down cleanly")
    
    def submit_batch(self, observations_batch: torch.Tensor) -> 'InferenceFuture':
        """Submit observation batch for GPU inference"""
        future = InferenceFuture()
        
        try:
            self.inference_queue.put((observations_batch, future), timeout=1.0)
        except:
            # Queue full - return dummy actions
            print("Inference queue full, returning dummy actions")
            dummy_actions = torch.zeros(observations_batch.shape[0], self.action_dim)
            future.set_result(dummy_actions)
            
        return future
    
    def _gpu_inference_worker(self):
        """Dedicated GPU worker for continuous inference processing"""
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Ensure GPU context
        print("Starting GPU inference worker thread")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next batch with timeout
                try:
                    batch_item = self.inference_queue.get(timeout=0.1)
                except Empty:
                    continue
                    
                if batch_item is None:  # Shutdown sentinel
                    break
                    
                observations_batch, future = batch_item
                
                # Skip if future already cancelled
                if hasattr(future, 'cancelled') and future.cancelled:
                    continue
                
                start_time = time.time()
                
                try:
                    # Execute batched diffusion inference on GPU
                    if torch.cuda.is_available():
                        with torch.cuda.stream(self.inference_stream):
                            actions_batch = self._batched_diffusion_inference(observations_batch)
                    else:
                        actions_batch = self._batched_diffusion_inference(observations_batch)
                    
                    # Performance tracking
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.batch_sizes.append(observations_batch.shape[0])
                    
                    # Reset error count on success
                    self.error_count = 0
                    
                    # Set result
                    future.set_result(actions_batch)
                    
                except Exception as e:
                    self.error_count += 1
                    print(f"GPU inference error #{self.error_count}: {e}")
                    traceback.print_exc()
                    
                    if self.error_count >= self.max_errors:
                        print(f"Too many errors ({self.error_count}), shutting down GPU worker")
                        future.set_error(RuntimeError(f"GPU worker failed after {self.error_count} errors"))
                        break
                    
                    # Set error on future
                    future.set_error(e)
                    
            except Exception as e:
                print(f"Unexpected error in GPU worker: {e}")
                traceback.print_exc()
                
        print("GPU inference worker stopped")
        
    def _batched_diffusion_inference(self, observations_batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorized diffusion inference across entire observation batch
        Core optimization: Single GPU call for all environments
        """
        batch_size = observations_batch.shape[0]
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision and torch.cuda.is_available():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        actions_batch = self._inference_impl(observations_batch, batch_size)
                else:
                    actions_batch = self._inference_impl(observations_batch, batch_size)
                    
            return actions_batch.float()
            
        except Exception as e:
            print(f"Error in batched diffusion inference: {e}")
            # Return random actions as fallback
            return torch.randn(batch_size, self.action_dim, device=observations_batch.device) * 0.1
     
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
            if torch.isnan(z_batch).any() or torch.isinf(z_batch).any():
                print(f"NaN/Inf detected at diffusion step {step}, reinitializing")
                z_batch = torch.randn_like(z_batch) * 0.1
        
        return z_batch


class InferenceFuture:
    """Thread-safe future for asynchronous inference results"""
    
    def __init__(self):
        self.result = None
        self.error = None
        self.ready = Event()
        self.cancelled = False  # Track cancellation state

    def set_result(self, result):
        if not self.cancelled:
            self.result = result
            self.ready.set()
    
    def set_error(self, error):
        if not self.cancelled:
            self.error = error
            self.ready.set()

    def cancel(self):
        """Mark this future as cancelled"""
        self.cancelled = True
        self.ready.set()
    
    def get(self, timeout=None):
        if self.ready.wait(timeout):
            if self.cancelled:
                raise RuntimeError("Inference was cancelled")
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
                                         thread_name_prefix=f'env-{self.worker_id}',
                                         mp_context=mp.get_context('spawn'))
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
    
    def close(self):
        """Clean up worker resources"""
        try:
            if self.env is not None:
                self.env.close()
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
        except Exception as e:
            print(f"Error in worker {self.worker_id} cleanup: {e}")   

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