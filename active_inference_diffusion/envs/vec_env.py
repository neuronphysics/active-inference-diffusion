"""
Custom implementation of vectorized environments with subprocess workers
Corrected version addressing all identified issues
"""

import multiprocessing as mp
from multiprocessing import Pipe, Process
from multiprocessing.sharedctypes import RawArray
import numpy as np
import gymnasium as gym
from typing import List, Callable, Tuple, Dict, Any, Optional, Union
import cloudpickle
import time
import signal
import sys
from enum import Enum
import ctypes
from abc import ABC, abstractmethod


class Commands(Enum):
    """Commands that can be sent to worker processes"""
    RESET = 'reset'
    STEP = 'step'
    CLOSE = 'close'
    GET_ATTR = 'get_attr'
    SET_ATTR = 'set_attr'
    SEED = 'seed'
    RENDER = 'render'


class CloudpickleWrapper:
    """
    Wrapper that uses cloudpickle to serialize contents
    This is necessary because standard pickle can't handle many Python objects
    like lambda functions or locally defined classes
    """
    def __init__(self, obj):
        self.obj = obj
    
    def __getstate__(self):
        return cloudpickle.dumps(self.obj)
    
    def __setstate__(self, state):
        self.obj = cloudpickle.loads(state)


def worker_process(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    worker_id: int,
    shared_memory: Optional[Dict[str, Any]] = None
) -> None:
    """
    The worker process that runs a single environment
    
    This function runs in a separate process and handles all environment
    interactions. It receives commands through the pipe and sends results back.
    
    Args:
        remote: The worker's end of the communication pipe
        parent_remote: The parent's end (closed immediately)
        env_fn_wrapper: Wrapped function that creates the environment
        worker_id: Unique identifier for this worker
        shared_memory: Optional shared memory buffers for observations
    """
    # Close the parent's end of the pipe in the worker
    parent_remote.close()
    
    # Create the environment
    env = env_fn_wrapper.obj()
    
    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        env.close()
        remote.close()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            # Wait for command from parent
            try:
                cmd, data = remote.recv()
            except EOFError:
                # Pipe was closed on the other end
                break
            
            if cmd == Commands.RESET:
                # Reset the environment
                kwargs = data or {}
                observation, info = env.reset(**kwargs)
                
                # For regular subprocess env, send the observation directly
                remote.send((observation, info))
                    
            elif cmd == Commands.STEP:
                # Take a step in the environment
                action = data
                observation, reward, terminated, truncated, info = env.step(action)
                remote.send((observation, reward, terminated, truncated, info))
                    
            elif cmd == Commands.SEED:
                # Seed the environment
                seed = data
                # Try new API first, fall back to old API
                try:
                    env.reset(seed=seed)
                    remote.send(seed)
                except TypeError:
                    if hasattr(env, 'seed'):
                        result = env.seed(seed)
                        remote.send(result)
                    else:
                        remote.send(None)
                        
            elif cmd == Commands.RENDER:
                # Render the environment
                kwargs = data or {}
                result = env.render(**kwargs) if hasattr(env, 'render') else None
                remote.send(result)
                
            elif cmd == Commands.GET_ATTR:
                # Get an attribute from the environment
                attr_name = data
                value = getattr(env, attr_name, None)
                remote.send(value)
                
            elif cmd == Commands.SET_ATTR:
                # Set an attribute on the environment
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)
                
            elif cmd == Commands.CLOSE:
                # Close the environment and exit
                env.close()
                remote.close()
                break
                
            else:
                raise ValueError(f"Unknown command: {cmd}")
                
    except Exception as e:
        # Log the error and close gracefully
        print(f"Worker {worker_id} error: {e}")
        env.close()
        remote.close()


def shmem_worker_process(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    worker_id: int,
    shared_memory_info: Dict[str, Any]
) -> None:
    """
    Worker process that uses shared memory for observations
    
    This is a specialized worker that writes observations directly to shared memory
    instead of sending them through the pipe, which is much more efficient for
    large observations like images.
    """
    parent_remote.close()
    env = env_fn_wrapper.obj()
    
    # Set up signal handling for graceful shutdown (FIX: Added signal handling)
    def signal_handler(signum, frame):
        env.close()
        remote.close()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Shared memory setup with proper byte offset calculation (FIX: Corrected offset calculation)
    obs_shape = shared_memory_info['obs_shape']
    dtype = shared_memory_info['obs_dtype']
    shared_array = shared_memory_info['shared_array']
    worker_offset = shared_memory_info['worker_offset']  # This is in elements, not bytes
    
    # Calculate the size of the observation in elements
    obs_size = int(np.prod(obs_shape))
    
    # Calculate byte offset - this is critical for correct memory access
    itemsize = np.dtype(dtype).itemsize
    offset_bytes = worker_offset * itemsize
    
    # Create a numpy view of this worker's portion of shared memory
    shared_obs = np.frombuffer(
        shared_array,
        dtype=dtype,
        count=obs_size,
        offset=offset_bytes  # Use byte offset, not element offset
    ).reshape(obs_shape)
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == Commands.RESET:
                kwargs = data or {}
                observation, info = env.reset(**kwargs)
                
                # Write observation to shared memory
                np.copyto(shared_obs, observation)
                
                # Send only info (observation is in shared memory)
                remote.send((None, info))
                
            elif cmd == Commands.STEP:
                action = data
                observation, reward, terminated, truncated, info = env.step(action)
                
                # Write observation to shared memory
                np.copyto(shared_obs, observation)
                
                # Send everything except observation
                remote.send((None, reward, terminated, truncated, info))
                
            # FIX: Added all missing command handlers
            elif cmd == Commands.SEED:
                seed = data
                try:
                    env.reset(seed=seed)
                    remote.send(seed)
                except TypeError:
                    if hasattr(env, 'seed'):
                        result = env.seed(seed)
                        remote.send(result)
                    else:
                        remote.send(None)
                        
            elif cmd == Commands.RENDER:
                kwargs = data or {}
                result = env.render(**kwargs) if hasattr(env, 'render') else None
                remote.send(result)
                
            elif cmd == Commands.GET_ATTR:
                attr_name = data
                value = getattr(env, attr_name, None)
                remote.send(value)
                
            elif cmd == Commands.SET_ATTR:
                attr_name, value = data
                setattr(env, attr_name, value)
                remote.send(None)
                
            elif cmd == Commands.CLOSE:
                env.close()
                remote.close()
                break
                
            else:
                raise ValueError(f"Unknown command: {cmd}")
                
    except Exception as e:
        print(f"Shared memory worker {worker_id} error: {e}")
        env.close()
        remote.close()


class BaseVectorEnv(ABC):
    """
    Base class for vectorized environments
    
    This provides the common interface and functionality for both
    SubprocVectorEnv and ShmemVectorEnv
    """
    
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = None
    ):
        self.num_envs = len(env_fns)
        self.closed = False
        
        # Set multiprocessing start method
        if start_method is not None:
            mp.set_start_method(start_method, force=True)
        
        # These will be set by subclasses
        self.workers = []
        self.remotes = []
        
        # Cache for environment properties
        self._observation_space = None
        self._action_space = None
        self._metadata = None
        
    @property
    def observation_space(self) -> gym.Space:
        """Get the observation space of a single environment"""
        if self._observation_space is None:
            self._observation_space = self.get_attr('observation_space')[0]
        return self._observation_space
    
    @property
    def action_space(self) -> gym.Space:
        """Get the action space of a single environment"""
        if self._action_space is None:
            self._action_space = self.get_attr('action_space')[0]
        return self._action_space
    
    @property
    def single_observation_space(self) -> gym.Space:
        """Alias for compatibility with existing code"""
        return self.observation_space
    
    @property
    def single_action_space(self) -> gym.Space:
        """Alias for compatibility with existing code"""
        return self.action_space
    
    def _assert_not_closed(self):
        """Check that the environments haven't been closed"""
        assert not self.closed, "Vectorized environment has been closed"
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments and return initial observations"""
        pass
    
    @abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Take a step in all environments"""
        pass
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get an attribute from the specified environments"""
        self._assert_not_closed()
        
        if indices is None:
            indices = list(range(self.num_envs))
        
        for i in indices:
            self.remotes[i].send((Commands.GET_ATTR, attr_name))
        
        results = []
        for i in indices:
            results.append(self.remotes[i].recv())
        
        return results
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """Set an attribute in the specified environments"""
        self._assert_not_closed()
        
        if indices is None:
            indices = list(range(self.num_envs))
        
        for i in indices:
            self.remotes[i].send((Commands.SET_ATTR, (attr_name, value)))
        
        # Wait for confirmation
        for i in indices:
            self.remotes[i].recv()
    
    def seed(self, seeds: Optional[Union[int, List[int]]] = None) -> List[Any]:
        """Seed the environments"""
        self._assert_not_closed()
        
        if seeds is None:
            seeds = [None] * self.num_envs
        elif isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        
        for remote, seed in zip(self.remotes, seeds):
            remote.send((Commands.SEED, seed))
        
        results = []
        for remote in self.remotes:
            results.append(remote.recv())
        
        return results
    
    def render(self, mode: str = 'human', **kwargs) -> List[Any]:
        """Render all environments"""
        self._assert_not_closed()
        
        for remote in self.remotes:
            remote.send((Commands.RENDER, kwargs))
        
        results = []
        for remote in self.remotes:
            results.append(remote.recv())
        
        return results
    
    def close(self):
        """Close all environments and worker processes"""
        if self.closed:
            return
        
        self.closed = True
        
        # Send close command to all workers
        for remote in self.remotes:
            try:
                remote.send((Commands.CLOSE, None))
            except:
                pass
        
        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        # Close all pipes
        for remote in self.remotes:
            remote.close()


class SharedMemoryArray:
    """
    A numpy array backed by shared memory
    
    This allows multiple processes to access the same memory without copying,
    which is crucial for performance with large observations like images.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        self.shape = shape
        self.dtype = dtype
        
        # Calculate total number of elements
        self.size = int(np.prod(shape))
        
        # Get the ctypes type for this numpy dtype (FIX: Better error handling)
        ctypes_type = self._get_ctypes_type(dtype)
        
        # Create shared memory buffer
        self.shared_array = RawArray(ctypes_type, self.size)
        
        # Create numpy view of shared memory
        self.np_array = np.frombuffer(
            self.shared_array,
            dtype=dtype
        ).reshape(shape)
    
    def _get_ctypes_type(self, dtype: np.dtype):
        """
        Map numpy dtype to ctypes type
        
        FIX: Now raises ValueError for unsupported types instead of 
        silently falling back to uint8
        """
        type_map = {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
            np.int32: ctypes.c_int32,
            np.int64: ctypes.c_int64,
            np.uint8: ctypes.c_uint8,
            np.bool_: ctypes.c_bool,
            np.int16: ctypes.c_int16,
            np.uint16: ctypes.c_uint16,
            np.uint32: ctypes.c_uint32,
            np.uint64: ctypes.c_uint64,
        }
        
        if dtype.type not in type_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Supported types are: {list(type_map.keys())}"
            )
        
        return type_map[dtype.type]
    
    def write(self, data: np.ndarray):
        """Write data to shared memory"""
        np.copyto(self.np_array, data)
    
    def read(self) -> np.ndarray:
        """Read data from shared memory (returns a copy for safety)"""
        return self.np_array.copy()
    
    # Note: Removed get_view() method as it's potentially unsafe
    # If you need direct access, use the read() method which returns a safe copy


class SubprocVectorEnv(BaseVectorEnv):
    """
    Vectorized environment that runs multiple environments in subprocesses
    
    This implementation uses pipes for communication between the main process
    and worker processes. It's suitable for most use cases and provides
    good performance with reasonable memory usage.
    """
    
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = 'spawn'
    ):
        super().__init__(env_fns, start_method)
        
        # Create pipes and processes for each environment
        for i, env_fn in enumerate(env_fns):
            parent_remote, worker_remote = Pipe()
            
            # Wrap the environment function for pickling
            wrapped_fn = CloudpickleWrapper(env_fn)
            
            # Create the worker process
            process = Process(
                target=worker_process,
                args=(worker_remote, parent_remote, wrapped_fn, i, None),
                daemon=True
            )
            process.start()
            
            # Close worker's end of pipe in parent
            worker_remote.close()
            
            self.workers.append(process)
            self.remotes.append(parent_remote)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments"""
        self._assert_not_closed()
        
        # Send reset command to all workers
        for remote in self.remotes:
            remote.send((Commands.RESET, kwargs))
        
        # Collect results
        observations = []
        infos = []
        
        for remote in self.remotes:
            obs, info = remote.recv()
            observations.append(obs)
            infos.append(info)
        
        # Stack observations into a single array
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Take a step in all environments"""
        self._assert_not_closed()
        
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"
        
        # Send actions to all workers
        for remote, action in zip(self.remotes, actions):
            remote.send((Commands.STEP, action))
        
        # Collect results
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, remote in enumerate(self.remotes):
            obs, reward, terminated, truncated, info = remote.recv()
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            
            # Add environment ID to info
            info['env_id'] = i
            infos.append(info)
        
        # Convert to numpy arrays
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )


class ShmemVectorEnv(BaseVectorEnv):
    """
    Vectorized environment using shared memory for observations
    
    This is optimized for environments with large observations (like images)
    where copying data between processes would be expensive. The observations
    are stored in shared memory that all processes can access directly.
    """
    
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = 'spawn'
    ):
        super().__init__(env_fns, start_method)
        
        # Create a dummy environment to get observation space
        dummy_env = env_fns[0]()
        obs_space = dummy_env.observation_space
        dummy_env.close()
        
        # Create shared memory for observations
        self.shared_obs = self._create_shared_memory(obs_space, self.num_envs)
        
        # Create pipes and processes
        for i, env_fn in enumerate(env_fns):
            parent_remote, worker_remote = Pipe()
            wrapped_fn = CloudpickleWrapper(env_fn)
            
            # Pass shared memory info to worker
            shared_memory_info = {
                'obs_shape': obs_space.shape,
                'obs_dtype': obs_space.dtype,
                'shared_array': self.shared_obs.shared_array,
                'worker_offset': i * int(np.prod(obs_space.shape))  # Element offset
            }
            
            process = Process(
                target=shmem_worker_process,
                args=(worker_remote, parent_remote, wrapped_fn, i, shared_memory_info),
                daemon=True
            )
            process.start()
            
            worker_remote.close()
            self.workers.append(process)
            self.remotes.append(parent_remote)
        
        # Store observation metadata
        self.obs_shape = obs_space.shape
        self.obs_dtype = obs_space.dtype
    
    def _create_shared_memory(self, obs_space: gym.Space, num_envs: int) -> SharedMemoryArray:
        """Create shared memory array for all environments' observations"""
        total_shape = (num_envs,) + obs_space.shape
        return SharedMemoryArray(total_shape, obs_space.dtype)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments"""
        self._assert_not_closed()
        
        for remote in self.remotes:
            remote.send((Commands.RESET, kwargs))
        
        infos = []
        for remote in self.remotes:
            _, info = remote.recv()
            infos.append(info)
        
        # Read observations from shared memory
        observations = self.shared_obs.read()
        
        return observations, infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Take a step in all environments"""
        self._assert_not_closed()
        
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"
        
        for remote, action in zip(self.remotes, actions):
            remote.send((Commands.STEP, action))
        
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, remote in enumerate(self.remotes):
            _, reward, terminated, truncated, info = remote.recv()
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            
            # FIX: Add env_id to info for consistency with SubprocVectorEnv
            info['env_id'] = i
            infos.append(info)
        
        # Read observations from shared memory
        observations = self.shared_obs.read()
        
        return (
            observations,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )