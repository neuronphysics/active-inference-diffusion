# ===== active_inference_diffusion/utils/buffers.py =====
"""
Replay buffer implementations
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import lz4.frame
import pickle
import random

# ----------------------------- SumTree -----------------------------

class _SumTree:
    """Simple SumTree for Prioritized Replay (CPU numpy)."""
    def __init__(self, capacity: int):
        assert capacity > 0
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data_index = np.zeros(self.capacity, dtype=np.int32)  # leaf_slot -> buffer_pos
        self.write = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data_idx: int) -> None:
        """Insert new leaf at current write pointer (leaf_slot) and map to buffer_pos=data_idx."""
        t_idx = self.write + self.capacity - 1
        self.data_index[self.write] = int(data_idx)
        self.update(t_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, t_idx: int, priority: float) -> None:
        change = float(priority) - float(self.tree[t_idx])
        self.tree[t_idx] = float(priority)
        self._propagate(t_idx, change)

    def _propagate(self, t_idx: int, change: float) -> None:
        parent = (t_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, t_idx: int, s: float) -> int:
        left = 2 * t_idx + 1
        right = left + 1
        if left >= len(self.tree):
            return t_idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def get(self, s: float) -> Tuple[int, float, int]:
        """Return (tree_index, priority_value, buffer_pos)."""
        t_idx = self._retrieve(0, s)
        leaf_slot = t_idx - self.capacity + 1
        return t_idx, float(self.tree[t_idx]), int(self.data_index[leaf_slot])


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


class PrioritizedSequenceReplayBuffer(SequenceReplayBuffer):
    """
    Prioritized sequence replay combining episodic storage with prioritized sampling.

    We keep a mapping buffer_pos -> (episode_id, step_in_episode). The SumTree stores
    leaves that map to buffer_pos (real storage positions in the parent). Sampling a
    leaf retrieves the buffer_pos; we look up the episode/time key to build a window.
    """
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        sequence_length: int = 10,
        overlap: int = 5,
        # PER
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        eps: float = 1e-6,
    ):
        super().__init__(capacity, obs_shape, action_dim, device, sequence_length, overlap)

        # PER hyperparameters
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_frames = int(beta_frames)
        self.eps = float(eps)
        self.frame_count = 0
        self.max_priority = 1.0

        # Tree with leaf->buffer_pos mapping
        self._tree = _SumTree(capacity)

        # buffer_pos -> (episode_id, step_in_episode); -1 means unknown
        self._pos_to_key: List[Optional[Tuple[int,int]]] = [None] * capacity

        # Episode-id tracking (monotonic ids) and O(1) id→index map
        self._next_episode_id: int = 0
        self._current_episode_id: Optional[int] = None
        self._episode_ids: List[int] = []          # index-aligned with self.episodes
        self._id_to_index: Dict[int, int] = {}     # ep_id -> current index in self.episodes

    # ------------------------ episode/key bookkeeping ------------------------

    def _on_episode_started(self) -> None:
        self._current_episode_id = self._next_episode_id
        self._next_episode_id += 1

    def _on_episode_finished(self) -> None:
        """Call when parent has appended a new episode to self.episodes (after add)."""
        if self._current_episode_id is None:
            return
        # Append ID and update dict
        self._episode_ids.append(self._current_episode_id)
        self._id_to_index[self._current_episode_id] = len(self._episode_ids) - 1

        # Keep in sync with parent's possible eviction policy:
        # If parent popped from the left (older episode), drop our leftmost id too.
        max_eps = self.capacity // self.sequence_length
        while len(self._episode_ids) > max_eps and len(self.episodes) > 0:
            popped_id = self._episode_ids.pop(0)
            self._id_to_index.pop(popped_id, None)
            # Shift remaining indices down by 1
            for eid in list(self._id_to_index.keys()):
                self._id_to_index[eid] -= 1
                if self._id_to_index[eid] < 0:
                    # defensive; should not happen
                    self._id_to_index.pop(eid, None)

        self._current_episode_id = None

    def _episode_index_from_id(self, ep_id: int) -> Optional[int]:
        return self._id_to_index.get(ep_id, None)

    # ------------------------------ API --------------------------------------

    def add(self, obs, action, reward, next_obs, done, frame_idx: int = 0, prev_action=None, priority: Optional[float] = None):
        """
        Add one environment step.

        FIXED: We use the parent's actual buffer position for the new write as our SumTree data_idx.
        We capture `buffer_pos = self.pos` BEFORE calling super().add(...), since most ring buffers
        write to `pos` then increment it.
        """
        episodes_before = len(self.episodes)

        # Start-of-episode detection
        episode_was_empty = (len(self.current_episode) == 0)
        if episode_was_empty:
            self._on_episode_started()

        # Capture mapping BEFORE push
        step_in_episode = len(self.current_episode)
        current_ep = self._current_episode_id if self._current_episode_id is not None else -1

        # Buffer position where parent will write this step (ring index)
        buffer_pos = int(self.pos)

        # Delegate to parent
        super().add(obs, action, reward, next_obs, done, frame_idx, prev_action)

        # Record mapping from buffer position to episode/time key
        self._pos_to_key[buffer_pos] = (current_ep, step_in_episode)

        # Assign initial priority and insert leaf (SumTree maps leaf_slot -> buffer_pos)
        base_pr = self.max_priority if priority is None else float(priority)
        self._tree.add(base_pr ** self.alpha, buffer_pos)
        self.max_priority = max(self.max_priority, base_pr)

        # Episode finish bookkeeping (robust)
        if len(self.episodes) > episodes_before:
            self._on_episode_finished()

    def sample_sequences(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample sequences via per-step PER while respecting episode boundaries."""
        if self._tree.n_entries == 0:
            return None

        # Anneal beta
        self.frame_count += 1
        beta = min(self.beta_end, self.beta_start + (self.beta_end - self.beta_start) * (self.frame_count / self.beta_frames))

        sequences = {
            'observations': [],
            'actions': [],
            'prev_actions': [],
            'rewards': [],
            'dones': [],
            'lengths': [],
            'frame_indices': [],
            'tree_indices': [],
        }
        priorities = []

        segment = max(self._tree.total, 1e-12) / batch_size

        for i in range(batch_size):
            # Try twice to avoid stale mappings
            for attempt in range(2):
                a, b = segment * i, segment * (i + 1)
                t_idx, p, buffer_pos = self._tree.get(random.uniform(a, b))
                key = self._pos_to_key[buffer_pos]

                if key is None:
                    # stale/overwritten; demote and retry
                    self._tree.update(t_idx, self.eps)
                    continue

                ep_id, step_idx = key
                epi_idx = self._episode_index_from_id(ep_id)
                if epi_idx is None or not (0 <= epi_idx < len(self.episodes)):
                    # episode evicted; demote and retry
                    self._tree.update(t_idx, self.eps)
                    continue

                episode = self.episodes[epi_idx]
                ep_len = len(episode)
                if ep_len == 0:
                    self._tree.update(t_idx, self.eps)
                    continue

                # Build window [seq_start, seq_end) centered if possible
                T = self.sequence_length
                half = T // 2
                seq_start = max(0, min(step_idx - half, ep_len - T))
                seq_end = min(ep_len, seq_start + T)
                if seq_end - seq_start < T:
                    seq_start = max(0, seq_end - T)
                subseq = episode[seq_start:seq_end]

                # Extract fields (parent API)
                obs_seq = [st['obs'] for st in subseq]
                if len(subseq) > 1:
                    act_seq  = [st['action'] for st in subseq[:-1]]
                    prev_seq = [st.get('prev_action', np.zeros_like(act_seq[0])) for st in subseq[:-1]]
                    rew_seq  = [st['reward'] for st in subseq[:-1]]
                    done_seq = [bool(st.get('done') or st.get('truncated')) for st in subseq[:-1]]
                else:
                    act_seq  = [subseq[0]['action']]
                    prev_seq = [subseq[0].get('prev_action', np.zeros_like(act_seq[0]))]
                    rew_seq  = [subseq[0]['reward']]
                    done_seq = [bool(subseq[0].get('done') or subseq[0].get('truncated'))]

                frame_idx_seq = [int(st['frame_idx']) for st in subseq]

                # Pad to fixed length
                actual_len = len(obs_seq)
                if actual_len < T:
                    pad_len = T - actual_len
                    obs_seq.extend([np.zeros_like(obs_seq[0])] * pad_len)
                    pad_acts = (T - 1) - len(act_seq)
                    if pad_acts > 0:
                        act_seq.extend([np.zeros_like(act_seq[0])] * pad_acts)
                        prev_seq.extend([np.zeros_like(prev_seq[0])] * pad_acts)
                        rew_seq.extend([0.0] * pad_acts)
                        done_seq.extend([True] * pad_acts)
                    frame_idx_seq.extend([0] * pad_len)

                # Record
                sequences['observations'].append(obs_seq)
                sequences['actions'].append(act_seq[:T-1])
                sequences['prev_actions'].append(prev_seq[:T-1])
                sequences['rewards'].append(rew_seq[:T-1])
                sequences['dones'].append(done_seq[:T-1])
                sequences['frame_indices'].append(frame_idx_seq[:T])
                sequences['lengths'].append(actual_len)
                sequences['tree_indices'].append(t_idx)
                priorities.append(p)
                break
            else:
                # very rare: fallback random sequence
                ridx = np.random.randint(0, len(self.episodes))
                episode = self.episodes[ridx]
                subseq = episode[:min(len(episode), self.sequence_length)]
                obs_seq = [st['obs'] for st in subseq]
                if len(subseq) > 1:
                    act_seq  = [st['action'] for st in subseq[:-1]]
                    prev_seq = [st.get('prev_action', np.zeros_like(act_seq[0])) for st in subseq[:-1]]
                    rew_seq  = [st['reward'] for st in subseq[:-1]]
                    done_seq = [bool(st.get('done') or st.get('truncated')) for st in subseq[:-1]]
                else:
                    act_seq  = [np.zeros_like(episode[0]['action'])]
                    prev_seq = [np.zeros_like(act_seq[0])]
                    rew_seq  = [0.0]
                    done_seq = [True]
                frame_idx_seq = [int(st['frame_idx']) for st in subseq]
                actual_len = len(obs_seq)
                if actual_len < self.sequence_length:
                    pad_len = self.sequence_length - actual_len
                    obs_seq.extend([np.zeros_like(obs_seq[0])] * pad_len)
                    pad_acts = (self.sequence_length - 1) - len(act_seq)
                    if pad_acts > 0:
                        act_seq.extend([np.zeros_like(act_seq[0])] * pad_acts)
                        prev_seq.extend([np.zeros_like(prev_seq[0])] * pad_acts)
                        rew_seq.extend([0.0] * pad_acts)
                        done_seq.extend([True] * pad_acts)
                    frame_idx_seq.extend([0] * pad_len)

                sequences['observations'].append(obs_seq)
                sequences['actions'].append(act_seq[:self.sequence_length-1])
                sequences['prev_actions'].append(prev_seq[:self.sequence_length-1])
                sequences['rewards'].append(rew_seq[:self.sequence_length-1])
                sequences['dones'].append(done_seq[:self.sequence_length-1])
                sequences['frame_indices'].append(frame_idx_seq[:self.sequence_length])
                sequences['lengths'].append(actual_len)
                sequences['tree_indices'].append(0)
                priorities.append(self.eps)

        # IS weights
        p_tot = max(self._tree.total, 1e-12)
        probs = np.asarray(priorities, dtype=np.float64) / p_tot
        N = max(self._tree.n_entries, 1)
        is_w = np.power(N * probs + 1e-12, -beta)
        is_w /= is_w.max()

        # Pack tensors on device
        out = {
            'observations':  torch.tensor(np.array(sequences['observations']), dtype=torch.float32, device=self.device),
            'actions':       torch.tensor(np.array(sequences['actions']),      dtype=torch.float32, device=self.device),
            'prev_actions':  torch.tensor(np.array(sequences['prev_actions']), dtype=torch.float32, device=self.device),
            'rewards':       torch.tensor(np.array(sequences['rewards']),      dtype=torch.float32, device=self.device),
            'dones':         torch.tensor(np.array(sequences['dones']),        dtype=torch.bool,    device=self.device),
            'lengths':       torch.tensor(sequences['lengths'],                dtype=torch.long,    device=self.device),
            'frame_indices': torch.tensor(np.array(sequences['frame_indices']),dtype=torch.long,    device=self.device),
            'tree_indices':  torch.tensor(sequences['tree_indices'],           dtype=torch.long,    device=self.device),
            'is_weights':    torch.tensor(is_w,                                 dtype=torch.float32, device=self.device),
        }
        # Alias for agent code
        out['frame_idx'] = out['frame_indices']
        return out

    @torch.no_grad()
    def update_priorities(self, tree_indices: torch.Tensor, priorities: torch.Tensor, *, already_alpha: bool=False) -> None:
        """
        Update priorities for sampled sequences.
        Expects RAW priorities (|td| + normalized ELBO + normalized dynamics).
        If you pass already-α-transformed priorities, set already_alpha=True.
        """
        if isinstance(tree_indices, torch.Tensor):
            tree_indices = tree_indices.detach().cpu().numpy()
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for t_idx, pr in zip(tree_indices, priorities):
            pr = max(float(pr), self.eps)
            pr_tree = pr if already_alpha else (pr ** self.alpha)
            self.max_priority = max(self.max_priority, pr)
            self._tree.update(int(t_idx), pr_tree)

    # ------------------------ helpers for priorities --------------------------

    @staticmethod
    def compute_sequence_priority(
        td_errors: Optional[torch.Tensor] = None,
        elbo_losses: Optional[torch.Tensor] = None,
        dynamics_losses: Optional[torch.Tensor] = None,
        weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Combine sequence-level errors into RAW priorities (no alpha here).
        Accepts either [B] or [B, T] tensors and aggregates across time.
        """
        xs = [x for x in (td_errors, elbo_losses, dynamics_losses) if x is not None]
        if not xs:
            raise ValueError("At least one of td_errors, elbo_losses, dynamics_losses must be provided.")
        device = xs[0].device
        B = xs[0].shape[0]
        pr = torch.zeros(B, device=device, dtype=torch.float32)

        if td_errors is not None:
            td = td_errors.abs()
            if td.dim() == 2:
                td = td.max(dim=1)[0]   # max over time
            pr = pr + weights[0] * td

        if elbo_losses is not None:
            el = elbo_losses.abs()
            if el.dim() == 2:
                el = el.mean(dim=1)     # mean over time
            el = el / (1.0 + el)        # squash
            pr = pr + weights[1] * el

        if dynamics_losses is not None:
            dy = dynamics_losses.abs()
            if dy.dim() == 2:
                dy = dy.mean(dim=1)
            dy = dy / (1.0 + dy)
            pr = pr + weights[2] * dy

        return pr + eps