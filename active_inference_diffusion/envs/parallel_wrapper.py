"""
Parallel Data Collection for Active Inference
Fixed version with proper CUDA handling for multiprocessing
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Process, Pipe
import gymnasium as gym
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import cloudpickle
import os

from active_inference_diffusion.agents.pixel_agent import DiffusionPixelAgent
from active_inference_diffusion.agents.state_agent import DiffusionStateAgent
from active_inference_diffusion.utils.buffers import ReplayBuffer
from active_inference_diffusion.configs.config import TrainingConfig


class CloudpickleWrapper:
    """Serialization wrapper for multiprocessing"""
    def __init__(self, x):
        self.x = x
    
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def env_worker(remote, parent_remote, env_fn_wrapper, agent_state_dict, agent_class, agent_config, device):
    """
    Worker process that runs environment and agent
    
    This worker creates a fresh agent from state dict to avoid CUDA context issues.
    The agent runs on CPU in the worker process.
    """
    parent_remote.close()
    
    # Force CPU in worker process
    device = 'cpu'
    torch.set_num_threads(1)  # Prevent thread explosion
    
    # Create environment
    env = env_fn_wrapper.x()
    
    # Create agent in this process (on CPU)
    agent = None
    if agent_state_dict is not None:
        try:
            # Recreate the agent configuration
            if agent_class == "DiffusionStateAgent":
                from active_inference_diffusion.agents.state_agent import DiffusionStateAgent
                from active_inference_diffusion.configs.config import ActiveInferenceConfig, TrainingConfig
                
                # Create a minimal agent for inference only
                agent = DiffusionStateAgent(env, agent_config['config'], agent_config['training_config'])
                
                # Load state dict with CPU mapping
                state_dict = torch.load(agent_state_dict, map_location='cpu')
                agent.active_inference.load_state_dict(state_dict['active_inference_state'])
                
            elif agent_class == "DiffusionPixelAgent":
                from active_inference_diffusion.agents.pixel_agent import DiffusionPixelAgent
                from active_inference_diffusion.configs.config import ActiveInferenceConfig, TrainingConfig, PixelObservationConfig
                
                agent = DiffusionPixelAgent(
                    env, 
                    agent_config['config'], 
                    agent_config['training_config'],
                    agent_config['pixel_config']
                )
                
                # Load state dict with CPU mapping
                state_dict = torch.load(agent_state_dict, map_location='cpu')
                agent.active_inference.load_state_dict(state_dict['active_inference_state'])
                if hasattr(agent, 'encoder'):
                    agent.encoder.load_state_dict(state_dict['encoder_state'])
            
            # Ensure agent is on CPU and in eval mode
            agent.device = torch.device('cpu')
            agent.active_inference = agent.active_inference.to('cpu')
            agent.active_inference.eval()
            
        except Exception as e:
            print(f"Failed to create agent in worker: {e}")
            agent = None
    
    while True:
        try:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # Execute environment step
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                obs, info = env.reset()
                remote.send((obs, info))
                
            elif cmd == 'act':
                # Agent selects action
                if agent is not None:
                    obs = data
                    with torch.no_grad():
                        action, action_info = agent.act(obs, deterministic=False)
                    remote.send((action, action_info))
                else:
                    # Random action if no agent
                    action = env.action_space.sample()
                    remote.send((action, {}))
                    
            elif cmd == 'render':
                if data == 'rgb_array':
                    frame = env.render()
                    remote.send(frame)
                else:
                    env.render()
                    
            elif cmd == 'close':
                env.close()
                remote.close()
                break
                
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
                
        except Exception as e:
            print(f"Worker error: {e}")
            import traceback
            traceback.print_exc()
            remote.send(('error', str(e)))


class ParallelDataCollector:
    """
    Parallel data collection for Active Inference agents
    
    This class manages multiple environment instances running in parallel,
    each with their own copy of the agent for action selection.
    """
    
    def __init__(
        self,
        env_fn,
        agent: Union[DiffusionStateAgent, DiffusionPixelAgent],
        num_envs: int = 4,
        device: str = "cuda"
    ):
        """
        Initialize parallel data collector
        
        Args:
            env_fn: Function that creates an environment
            agent: The active inference agent (will be copied to each process)
            num_envs: Number of parallel environments
            device: Device for main agent (workers always use CPU)
        """
        # Set spawn method for CUDA compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        self.num_envs = num_envs
        self.device = device
        self.agent = agent
        self.closed = False
        
        # Save agent state dict and configuration
        import tempfile
        self.agent_checkpoint = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        
        # Determine agent class and save minimal state
        if isinstance(agent, DiffusionStateAgent):
            agent_class = "DiffusionStateAgent"
            agent_config = {
                'config': agent.config,
                'training_config': agent.training_config
            }
        else:
            agent_class = "DiffusionPixelAgent"
            agent_config = {
                'config': agent.config,
                'training_config': agent.training_config,
                'pixel_config': agent.pixel_config
            }
        
        # Save only the necessary state dicts
        checkpoint = {
            'active_inference_state': agent.active_inference.state_dict(),
        }
        if hasattr(agent, 'encoder'):
            checkpoint['encoder_state'] = agent.encoder.state_dict()
        
        torch.save(checkpoint, self.agent_checkpoint.name)
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        
        # Start worker processes with spawn method
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = mp.Process(
                target=env_worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn), 
                      self.agent_checkpoint.name, agent_class, agent_config, 'cpu')
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
            
        # Close worker remotes in main process
        for remote in self.work_remotes:
            remote.close()
            
        # Get environment spaces
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        
        # Initialize current observations
        self._reset_all()
        
    def _reset_all(self):
        """Reset all environments"""
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        self.observations = [result[0] for result in results]
        self.infos = [result[1] for result in results]
        
    def collect_steps(self, num_steps: int, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """
        Collect multiple steps of experience in parallel
        
        Args:
            num_steps: Total number of steps to collect across all environments
            replay_buffer: Buffer to store transitions
            
        Returns:
            Dictionary with collection statistics
        """
        steps_collected = 0
        episodes_completed = 0
        total_rewards = []
        episode_rewards = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        
        while steps_collected < num_steps:
            # Get actions from agent for all environments
            actions = []
            action_infos = []
            
            for i, obs in enumerate(self.observations):
                self.remotes[i].send(('act', obs))
            
            for i, remote in enumerate(self.remotes):
                action, info = remote.recv()
                actions.append(action)
                action_infos.append(info)
            
            # Step all environments
            for i, (remote, action) in enumerate(zip(self.remotes, actions)):
                remote.send(('step', action))
            
            # Collect results
            for i, remote in enumerate(self.remotes):
                next_obs, reward, done, info = remote.recv()
                
                # Add to replay buffer
                replay_buffer.add(
                    self.observations[i],
                    actions[i],
                    reward,
                    next_obs,
                    done
                )
                
                # Update statistics
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                steps_collected += 1
                
                if done:
                    # Episode completed
                    total_rewards.append(episode_rewards[i])
                    episodes_completed += 1
                    
                    # Reset this environment
                    remote.send(('reset', None))
                    obs, info = remote.recv()
                    self.observations[i] = obs
                    
                    # Reset episode stats
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                else:
                    self.observations[i] = next_obs
        
        # Compute statistics
        stats = {
            'steps_collected': steps_collected,
            'episodes_completed': episodes_completed,
        }
        
        if total_rewards:
            stats.update({
                'mean_episode_reward': np.mean(total_rewards),
                'std_episode_reward': np.std(total_rewards),
                'min_episode_reward': np.min(total_rewards),
                'max_episode_reward': np.max(total_rewards),
            })
            
        return stats
    
    def render(self, mode='rgb_array'):
        """Render all environments"""
        if mode == 'rgb_array':
            frames = []
            for remote in self.remotes:
                remote.send(('render', mode))
            for remote in self.remotes:
                frame = remote.recv()
                frames.append(frame)
            return np.stack(frames)
        else:
            for remote in self.remotes:
                remote.send(('render', mode))
    
    def close(self):
        """Clean up resources"""
        if self.closed:
            return
            
        # Close all environments
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except:
                pass
            
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
            
        # Clean up agent checkpoint
        try:
            os.unlink(self.agent_checkpoint.name)
        except:
            pass
        
        self.closed = True
    
    def __del__(self):
        self.close()


def create_parallel_collector(
    env_name: str,
    agent: Union[DiffusionStateAgent, DiffusionPixelAgent],
    num_envs: int = 4,
    use_pixels: bool = False,
    seed: int = 0
) -> ParallelDataCollector:
    """
    Create parallel data collector with same environment setup as training
    """
    from active_inference_diffusion.envs.wrappers import NormalizeObservation, ActionRepeat
    from active_inference_diffusion.envs.pixel_wrappers import make_pixel_mujoco
    
    def make_env(env_seed):
        def _init():
            if use_pixels:
                env = make_pixel_mujoco(
                    env_name,
                    width=84,
                    height=84,
                    frame_stack=3,
                    action_repeat=2,
                    seed=env_seed
                )
            else:
                env = gym.make(env_name)
                env = NormalizeObservation(env)
                env = ActionRepeat(env, repeat=2)
                env.reset(seed=env_seed)
                env.action_space.seed(env_seed)
                env.observation_space.seed(env_seed)
            return env
        return _init
    
    # Create collector with different seeds for each environment
    collector = ParallelDataCollector(
        env_fn=make_env(seed),
        agent=agent,
        num_envs=num_envs,
        device=str(agent.device)
    )
    
    return collector


# Modified training loop integration
def train_with_parallel_collection(
    agent: Union[DiffusionStateAgent, DiffusionPixelAgent],
    training_config: TrainingConfig,
    env_name: str,
    use_pixels: bool = False,
    num_parallel_envs: int = TrainingConfig.num_parallel_envs,
    collection_steps: int = 1000
) -> Dict[str, float]:
    """
    Enhanced training loop with parallel data collection
    
    This replaces the single environment interaction in the original training loop
    with efficient parallel collection.
    """
    # Create parallel collector
    collector = create_parallel_collector(
        env_name=env_name,
        agent=agent,
        num_envs=num_parallel_envs,
        use_pixels=use_pixels
    )
    
    try:
        # Collect data in parallel
        print(f"Collecting {collection_steps} steps across {num_parallel_envs} environments...")
        
        collection_stats = collector.collect_steps(
            num_steps=collection_steps,
            replay_buffer=agent.replay_buffer
        )
        
        print(f"Collected {collection_stats['steps_collected']} steps, "
              f"completed {collection_stats['episodes_completed']} episodes")
        
        # Training phase (same as original)
        train_metrics = {}
        if len(agent.replay_buffer) >= training_config.learning_starts:
            for _ in range(training_config.gradient_steps * collection_steps):
                metrics = agent.train_step()
                train_metrics.update(metrics)
        
        # Update exploration
        agent.update_exploration()
        
        # Combine stats
        all_metrics = {**collection_stats, **train_metrics}
        
        return all_metrics
        
    finally:
        collector.close()