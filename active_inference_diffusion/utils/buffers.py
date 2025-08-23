# ===== active_inference_diffusion/utils/buffers.py =====
"""
Replay buffer implementations
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import lz4.frame
import pickle

class ReplayBuffer:
    """
    Efficient replay buffer with optional compression
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        optimize_memory: bool = False
    ):
        self.capacity = capacity
        self.device = device
        self.optimize_memory = optimize_memory
        self.obs_shape = obs_shape
        self.pos = 0
        self.size = 0
        self.cleanup_count = 0
        self.cleanup_interval = 2000  # Cleanup every 2000 additions
        self.frame_idx = torch.zeros(capacity, dtype=torch.long, device=device)
        # Allocate memory
        if optimize_memory and len(obs_shape) == 3:  # Pixel observations
            self.observations = [None] * capacity
            self.next_observations = [None] * capacity
            self.compress = True
            self.dtype = np.uint8
        else:
            self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
            self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
            self.compress = False
            self.dtype = np.float32
            
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.prev_actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.bool)
        
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        frame_idx: int = 0,
        prev_action: Optional[np.ndarray] = None
    ):
        """Add transition to buffer"""
        if self.compress:
            # Ensure uint8 format for pixel observations
            if obs.dtype != np.uint8:
                if obs.max() <= 1.0:  # Normalized
                    obs = (obs * 255).astype(np.uint8)
                else:
                    obs = obs.astype(np.uint8)
            if next_obs.dtype != np.uint8:
                if next_obs.max() <= 1.0:
                    next_obs = (next_obs * 255).astype(np.uint8)
                else:
                    next_obs = next_obs.astype(np.uint8)
                    
            self.observations[self.pos] = self._compress(obs)
            self.next_observations[self.pos] = self._compress(next_obs)
        else:
            self.observations[self.pos] = torch.from_numpy(obs)
            self.next_observations[self.pos] = torch.from_numpy(next_obs)
            
        self.actions[self.pos] = torch.from_numpy(action)
        if prev_action is None:
            self.prev_actions[self.pos] = torch.zeros_like(self.actions[self.pos])
        else:
            self.prev_actions[self.pos] = torch.as_tensor(prev_action, dtype=torch.float32)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = bool(done)
        self.frame_idx[self.pos] = int(frame_idx)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.cleanup_count += 1
        if self.compress and self.cleanup_count >= self.cleanup_interval:
            # Perform cleanup to free memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cleanup_count = 0
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions"""
        indices = np.random.randint(0, self.size, batch_size)
        
        if self.compress:
            # Decompress observations
            obs_list = []
            next_obs_list = []
            
            for i in indices:
                obs = self._decompress(self.observations[i])
                next_obs = self._decompress(self.next_observations[i])

                obs_list.append(torch.from_numpy(obs).float() / 255.0)
                next_obs_list.append(torch.from_numpy(next_obs).float() / 255.0)
                
            obs = torch.stack(obs_list)
            next_obs = torch.stack(next_obs_list)
        else:
            obs = self.observations[indices]
            next_obs = self.next_observations[indices]
            
        return {
            'observations': obs,
            'actions': self.actions[indices],
            'prev_actions': self.prev_actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': next_obs,
            'dones': self.dones[indices],  # Already float
            'frame_idx': self.frame_idx[indices]
        }
        
    def _compress(self, data: np.ndarray) -> bytes:
        """Compress numpy array"""
        meta = {
            'shape': data.shape,
            'dtype': str(data.dtype)
        }
        # Convert to bytes if needed
        if data.dtype == np.uint8:
            data_bytes = data.tobytes()
        else:
            data_bytes = data.astype(np.float32).tobytes()
            
        # Pickle metadata and compress everything
        return lz4.frame.compress(pickle.dumps((meta, data_bytes)))
        
    def _decompress(self, compressed_data: bytes) -> np.ndarray:
        """Decompress to numpy array"""
        meta, data_bytes = pickle.loads(lz4.frame.decompress(compressed_data))
        
        # Reconstruct array with correct shape and dtype
        if meta['dtype'] == 'uint8':
            array = np.frombuffer(data_bytes, dtype=np.uint8).reshape(meta['shape'])
        else:
            array = np.frombuffer(data_bytes, dtype=np.float32).reshape(meta['shape'])
            
        return array
        
    def __len__(self):
        return self.size
    
class SequenceReplayBuffer(ReplayBuffer):
    """Replay buffer that stores sequences for LSTM training"""
    
    def __init__(self, capacity, obs_shape, action_dim, device, 
                 sequence_length=10, overlap=5):
        super().__init__(capacity, obs_shape, action_dim, device)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.episodes = []  # Store complete episodes
        self.current_episode = []

    def add(self, obs, action, reward, next_obs, done, frame_idx=0, prev_action=None):
        # Add to current episode
        self.current_episode.append({
            'obs': obs, 'action': action, 'reward': reward,
            'next_obs': next_obs, 'done': done, 'frame_idx': frame_idx,
            'prev_action': prev_action
        })
        
        # If episode ends, store it
        if done:
            if len(self.current_episode) > 1:
                self.episodes.append(self.current_episode)
                if len(self.episodes) > self.capacity // self.sequence_length:
                    self.episodes.pop(0)
            self.current_episode = []
        
        # Also add to regular buffer for standard sampling
        super().add(obs, action, reward, next_obs, done, frame_idx, prev_action=prev_action)
    
    def sample_sequences(self, batch_size):
        """Sample sequences with proper padding and masking"""
        if len(self.episodes) < batch_size:
            return None
            
        # Sample episodes
        sampled_episodes = np.random.choice(self.episodes, batch_size, replace=True)
        
        # Create padded sequences
        max_len = self.sequence_length
        sequences = {
            'observations': [],
            'actions': [],
            'prev_actions': [],
            'rewards': [],
            'dones': [],
            'lengths': [],
            'frame_indices': []
        }
        
        for episode in sampled_episodes:
            # Sample a subsequence from the episode
            ep_len = len(episode)
            if ep_len > max_len:
                start_idx = np.random.randint(0, ep_len - max_len + 1)
                subsequence = episode[start_idx:start_idx + max_len]
            else:
                subsequence = episode
                
            # Extract data
            obs_seq = [step['obs'] for step in subsequence]
            act_seq = [step['action'] for step in subsequence[:-1]]  # One less action
            prev_act_seq = [step.get('prev_action', np.zeros_like(act_seq[0])) 
                        for step in subsequence[:-1]]
            rew_seq = [step['reward'] for step in subsequence[:-1]]
            done_seq = [step['done'] or step.get('truncated') for step in subsequence[:-1]]
            frame_idx_seq = [step['frame_idx'] for step in subsequence]

            # Pad if necessary
            actual_len = len(obs_seq)
            if actual_len < max_len:
                # Pad with zeros
                pad_len = max_len - actual_len
                obs_seq.extend([np.zeros_like(obs_seq[0])] * pad_len)
                act_seq.extend([np.zeros_like(act_seq[0])] * (pad_len))
                prev_act_seq.extend([np.zeros_like(prev_act_seq[0])] * pad_len) 
                rew_seq.extend([0.0] * pad_len)
                done_seq.extend([True] * pad_len)  # Mark padded as done
                frame_idx_seq.extend([0] * pad_len)  # Zero frame indices for padding
            
            sequences['observations'].append(obs_seq)
            sequences['actions'].append(act_seq[:max_len-1])
            sequences['prev_actions'].append(prev_act_seq[:max_len-1])
            sequences['rewards'].append(rew_seq[:max_len-1])
            sequences['dones'].append(done_seq[:max_len-1])
            sequences['frame_indices'].append(frame_idx_seq[:max_len])
            sequences['lengths'].append(actual_len)
        
        # Convert to tensors
        return {
            'observations': torch.tensor(np.array(sequences['observations']), 
                                       dtype=torch.float32, device=self.device),
            'actions': torch.tensor(np.array(sequences['actions']), 
                                  dtype=torch.float32, device=self.device),
            'prev_actions': torch.tensor(np.array(sequences['prev_actions']), 
                                       dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(np.array(sequences['rewards']), 
                                  dtype=torch.float32, device=self.device),
            'dones': torch.tensor(np.array(sequences['dones']), 
                                dtype=torch.bool, device=self.device),
            'lengths': torch.tensor(sequences['lengths'], 
                                  dtype=torch.long, device=self.device),
            'frame_indices': torch.tensor(np.array(sequences['frame_indices']), 
                                       dtype=torch.long, device=self.device),
        }

