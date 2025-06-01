import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from typing import Optional, Tuple, Dict, Any
import warnings


class MuJoCoPixelObservationWrapper(gym.ObservationWrapper):
    """
    Converts MuJoCo environments from state observations to pixel observations.
    
    This wrapper intercepts the normal state observations and instead returns
    RGB images rendered from the MuJoCo camera.
    
    Args:
        env: The MuJoCo environment to wrap
        width: Width of rendered images (default: 84)
        height: Height of rendered images (default: 84)
        camera_name: Name of MuJoCo camera to use (None for default)
        frame_skip: Number of frames to skip between renders
        channels_first: If True, returns (C, H, W), else (H, W, C)
        normalize: If True, normalizes pixels to [0, 1] range
    """
    
    def __init__(
        self,
        env: gym.Env,
        width: int = 84,
        height: int = 84,
        camera_name: Optional[str] = None,
        frame_skip: int = 1,
        channels_first: bool = True,
        normalize: bool = False,
        device_id: int = -1  # -1 for CPU, 0+ for GPU
    ):
        super().__init__(env)
        
        # Validate that this is a MuJoCo environment
        if not hasattr(env.unwrapped, 'mujoco_renderer'):
            # For older gym versions or different MuJoCo wrappers
            if hasattr(env.unwrapped, 'sim'):
                # Legacy mujoco-py environments
                self._render_mode = 'mujoco_py'
            else:
                raise ValueError("Environment does not appear to be a MuJoCo environment")
        else:
            # Modern gymnasium MuJoCo environments
            self._render_mode = 'gymnasium'
            
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.frame_skip = frame_skip
        self.channels_first = channels_first
        self.normalize = normalize
        self.device_id = device_id
        
        # Frame counter for frame skipping
        self._frame_count = 0
        self._last_pixels = None
        
        # Update observation space
        shape = (3, height, width) if channels_first else (height, width, 3)
        dtype = np.float32 if normalize else np.uint8
        high = 1.0 if normalize else 255
        
        self.observation_space = Box(
            low=0,
            high=high,
            shape=shape,
            dtype=dtype
        )
        
        # Store original observation space for reference
        self._original_obs_space = env.observation_space
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Convert state observation to pixel observation.
        
        Args:
            obs: Original state observation (ignored)
            
        Returns:
            Pixel observation as numpy array
        """
        # Handle frame skipping
        self._frame_count += 1
        if self.frame_skip > 1 and self._frame_count % self.frame_skip != 0:
            if self._last_pixels is not None:
                return self._last_pixels
        
        # Render pixels based on environment type
        pixels = self._render_pixels()
        
        # Store for frame skipping
        self._last_pixels = pixels
        
        return pixels
    
    def _render_pixels(self) -> np.ndarray:
        """Render pixels using appropriate method based on environment type."""
        # Navigate to the base MuJoCo environment through wrapper chain
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        # Alternative unwrapped access for gymnasium environments
        if hasattr(self.env, 'unwrapped'):
            base_env = self.env.unwrapped
    
        if self._render_mode == 'gymnasium':
            # Modern gymnasium approach - use base environment's renderer
            if self.camera_name is not None:
                # Set camera before rendering
                if hasattr(base_env, 'mujoco_renderer') and hasattr(base_env.mujoco_renderer, 'camera_name'):
                    base_env.mujoco_renderer.camera_name = self.camera_name

            pixels = base_env.mujoco_renderer.render(
                render_mode='rgb_array',
                camera_name=self.camera_name,
                width=self.width,
                height=self.height
            )
        else:
            # Legacy mujoco-py approach
            if self.camera_name is None:
                pixels = base_env.sim.render(
                    width=self.width,
                    height=self.height,
                    mode='offscreen',
                    device_id=self.device_id
                )
            else:
                pixels = base_env.sim.render(
                    width=self.width,
                    height=self.height,
                    camera_name=self.camera_name,
                    mode='offscreen',
                    device_id=self.device_id
                )
    
        # Flip vertically (OpenGL convention)
        pixels = pixels[::-1, :, :]
    
        # Convert to channels-first if requested
        if self.channels_first and pixels.shape[2] == 3:
            pixels = np.transpose(pixels, (2, 0, 1))
    
        # Normalize if requested
        if self.normalize:
            pixels = pixels.astype(np.float32) / 255.0
        else:
            pixels = pixels.astype(np.uint8)
        
        return pixels
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return pixel observation."""
        obs, info = self.env.reset(**kwargs)
        self._frame_count = 0
        self._last_pixels = None
        
        # Add original state observation to info for debugging
        info['state_obs'] = obs
        
        return self.observation(obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and return pixel observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add original state observation to info
        info['state_obs'] = obs
        
        return self.observation(obs), reward, terminated, truncated, info


class MuJoCoPixelDictObservationWrapper(MuJoCoPixelObservationWrapper):
    """
    Extended wrapper that returns both pixels and state in a dict observation.
    
    Useful for methods that need both visual and proprioceptive information.
    """
    
    def __init__(self, env: gym.Env, state_key: str = 'state', pixel_key: str = 'pixels', **kwargs):
        self.state_key = state_key
        self.pixel_key = pixel_key
        super().__init__(env, **kwargs)
        
        # Update observation space to dict
        self.observation_space = gym.spaces.Dict({
            self.pixel_key: self.observation_space,
            self.state_key: self._original_obs_space
        })
    
    def observation(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Return both pixel and state observations."""
        pixels = super().observation(obs)
        return {
            self.pixel_key: pixels,
            self.state_key: obs
        }


class MultiCameraWrapper(gym.ObservationWrapper):
    """
    Wrapper that provides multiple camera views simultaneously.
    
    Useful for learning view-invariant representations.
    """
    
    def __init__(
        self,
        env: gym.Env,
        camera_configs: Dict[str, Dict[str, Any]],
        channels_first: bool = True,
        normalize: bool = False
    ):
        super().__init__(env)
        
        self.camera_configs = camera_configs
        self.channels_first = channels_first
        self.normalize = normalize
        
        # Create observation space
        spaces = {}
        for cam_name, config in camera_configs.items():
            shape = (3, config['height'], config['width']) if channels_first else (config['height'], config['width'], 3)
            dtype = np.float32 if normalize else np.uint8
            high = 1.0 if normalize else 255
            
            spaces[cam_name] = Box(low=0, high=high, shape=shape, dtype=dtype)
            
        self.observation_space = gym.spaces.Dict(spaces)
        
    def observation(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """Render from multiple cameras."""
        multi_cam_obs = {}
        
        for cam_name, config in self.camera_configs.items():
            # Create temporary wrapper for this camera
            temp_wrapper = MuJoCoPixelObservationWrapper(
                self.env,
                width=config['width'],
                height=config['height'],
                camera_name=cam_name,
                channels_first=self.channels_first,
                normalize=self.normalize
            )
            
            multi_cam_obs[cam_name] = temp_wrapper._render_pixels()
            
        return multi_cam_obs


# Convenience functions for common MuJoCo environments

from .wrappers import ActionRepeat  # Add this import at the top of the file

def make_pixel_mujoco(
    env_id: str,
    width: int = 84,
    height: int = 84,
    frame_stack: int = 3,
    action_repeat: int = 2,
    camera_name: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> gym.Env:
    """
    Create a pixel-based MuJoCo environment with common wrappers.
    
    This function orchestrates a hierarchical wrapper composition that transforms
    raw MuJoCo state-based environments into pixel-observation spaces suitable
    for visual control learning paradigms.
    
    Args:
        env_id: MuJoCo environment identifier
        width: Rendered observation width (pixels)
        height: Rendered observation height (pixels)
        frame_stack: Temporal context via frame concatenation
        action_repeat: Temporal action persistence factor
        camera_name: MuJoCo camera perspective identifier
        seed: Stochastic initialization seed
        **kwargs: Additional MuJoCoPixelObservationWrapper parameters
        
    Returns:
        Hierarchically wrapped environment with pixel observations
    """
    # Create base environment
    env = gym.make(env_id)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    # Add action repeat wrapper if requested
    # This implements temporal action persistence, reducing control frequency
    # while maintaining physics simulation fidelity
    if action_repeat > 1:
        env = ActionRepeat(env, repeat=action_repeat)  # Use your custom wrapper
    
    # Add pixel observation wrapper
    # Transforms proprioceptive state space to visual observation space
    env = MuJoCoPixelObservationWrapper(
        env,
        width=width,
        height=height,
        camera_name=camera_name,
        **kwargs
    )
    
    # Add frame stacking if requested
    # Provides temporal context through observation history
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
    
    return env