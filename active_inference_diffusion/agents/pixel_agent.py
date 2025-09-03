"""
Pixel-Based Active Inference Agent
Integrates visual encoding with diffusion-generated latent spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional, Any

from .base_agent import BaseActiveInferenceAgent
from ..core.active_inference import DiffusionActiveInference, EMAModel
from ..encoder.visual_encoders import RandomShiftAugmentation, DrQV2Encoder
from ..encoder.state_encoders import EncoderFactory
from ..utils.buffers import ReplayBuffer, SequenceReplayBuffer, PrioritizedSequenceReplayBuffer
from ..utils.util import _normalize_metrics
from ..configs.config import (
    ActiveInferenceConfig,
    PixelObservationConfig,
    TrainingConfig
)


class DiffusionPixelAgent(BaseActiveInferenceAgent):
    """
    Pixel-based agent using diffusion-generated latent active inference
    
    Key innovations:
    - Visual observations encoded to feature space before latent generation
    - Contrastive learning enhances visual representation quality
    - Seamless integration with diffusion-based belief updates
    """
    
    def __init__(
        self,
        env: gym.Env,
        config: ActiveInferenceConfig,
        training_config: TrainingConfig,
        pixel_config: PixelObservationConfig
    ):
        self.pixel_config = pixel_config
        super().__init__(env, config, training_config)
        
    def _setup_dimensions(self):
        """Setup dimensions for pixel observations"""
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Validate observation space
        if isinstance(self.obs_space, gym.spaces.Box):
            obs_shape = self.obs_space.shape
            if len(obs_shape) == 3:  # (C, H, W) or (H, W, C)
                self.frame_stack = 1
                self.obs_shape = obs_shape
                # Ensure channels-first format
                if obs_shape[-1] in [1, 3]:  # Likely (H, W, C)
                    self.obs_shape = (obs_shape[-1], obs_shape[0], obs_shape[1])
            elif len(obs_shape) == 4:  # (B, C, H, W) or (B, H, W, C)
                self.frame_stack = obs_shape[0]
                self.obs_shape = obs_shape[1:]
                print(f"Using frame stack: {self.frame_stack}")
            else:
                raise ValueError(f"Unexpected observation shape: {obs_shape}")
        else:
            raise ValueError(f"Unsupported observation space: {type(self.obs_space)}")
            
        # Validate action space
        if isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.action_space)}")
            
        # Update configuration
        self.config.action_dim = self.action_dim
        # Set observation dimension to encoder output dimension
        if self.pixel_config.pixel_observation:
            self.config.observation_dim = self.config.latent_dim
        else:
            self.config.observation_dim = self.obs_space[0]
        
    def _build_models(self):
        """Build visual encoding and diffusion active inference models"""
        # Visual encoder
        self.encoder = DrQV2Encoder(
            obs_shape=self.obs_shape,
            feature_dim=self.config.latent_dim,
            frame_stack=self.pixel_config.frame_stack,
            num_layers=self.pixel_config.num_layers,
            num_filters=32,
            use_spectral_norm=True,
            attention='global',
            checkpoint_trunk=True,  
            checkpoint_attention=True,
            checkpoint_head=False
            ).to(self.device)
        
        # Augmentation module
        self.augmentation = RandomShiftAugmentation(pad=self.pixel_config.random_shift_pad) if self.pixel_config.augmentation else None
        
        # Core diffusion active inference
        # Uses encoder output dimension as observation dimension
        self.active_inference = DiffusionActiveInference(
            observation_dim=self.config.observation_dim,
            action_dim=self.action_dim,
            latent_dim=self.config.latent_dim,
            config=self.config,
            pixel_shape=self.obs_shape if self.pixel_config.pixel_observation else None,
            shared_visual_encoder=self.encoder 
        )
        self.value_ema = EMAModel(self.active_inference.value_network, decay=0.9999, device=self.device)

        # Move all components to device
        self.encoder = self.encoder.to(self.device)
        self.active_inference = self.active_inference.to(self.device)
        
    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        frame_idx: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using visual encoding and diffusion active inference
        
        Pipeline: pixels -> features -> diffusion latents -> policy
        """
        # Process raw pixels
        obs_tensor = self._process_observation(observation)
        
        # Encode to feature space
        with torch.no_grad():
            if hasattr(self, 'encoder') and self.encoder is not None:
                self.encoder = self.encoder.to(self.device)

            encoded_obs = self.encode_observation(obs_tensor.to(self.device))
            if encoded_obs.dim() == 1:
               encoded_obs = encoded_obs.unsqueeze(0)
            
            # Use diffusion active inference with encoded features
            action_tensor, info = self.active_inference.act(
                encoded_obs.squeeze(0),  # Remove batch dimension
                deterministic=deterministic,
                raw_observation=obs_tensor,
                from_idx= frame_idx,
                actions=actions
            )
            
        # Convert to numpy
        action = action_tensor.cpu().numpy()
        # Handle different action shapes
        if action.ndim == 0:  # Scalar
            action = np.array([action])
        elif action.ndim == 2 and action.shape[0] == 1:  # [1, action_dim]
            action = action[0]
        elif action.ndim == 1:  # Already correct shape
            pass
        else:
            # Unexpected shape - try to flatten to 1D
            action = action.flatten()
        if hasattr(self, 'action_dim') and len(action) != self.action_dim:
            print(f"Warning: action shape {action.shape} doesn't match expected {self.action_dim}")
         
        # Add exploration noise if training
        if not deterministic and self.training and self.exploration_noise > 0:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        # Add encoding info
        info['encoded_obs_norm'] = encoded_obs.norm().item()
        
        return action, info
        
    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode pixel observation to feature space with augmentation"""
        if observation.device != next(self.encoder.parameters()).device:
            observation = observation.to(next(self.encoder.parameters()).device)
    
        # If observation is uint8, normalize to [0, 1]
        if observation.dtype == torch.uint8:
            observation = observation.float() / 255.0
 
        # Apply augmentation during training
        # Handle different input formats
        if observation.ndim == 5:  # (batch, frame_stack, C, H, W)
            batch_size, frame_stack, c, h, w = observation.shape
            # Reshape to (batch, frame_stack * C, H, W) for encoder
            observation = observation.view(batch_size, frame_stack * c, h, w)
        elif observation.ndim == 4 and hasattr(self, 'frame_stack') and self.frame_stack > 1:
            # Single sample with frame stack
            if observation.shape[0] == 1:  # Batch dimension
                # Check if second dimension is frame stack
                if observation.shape[1] == self.frame_stack:
                    batch_size = 1
                    observation = observation.view(batch_size, -1, 
                                             observation.shape[-2], 
                                             observation.shape[-1])
        elif observation.ndim == 3:  # Single image without batch
            observation = observation.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected observation shape: {observation.shape}")
    
        # Apply augmentation during training
        if self.augmentation is not None and self.training:
            observation = self.augmentation(observation)
        
        # Encode to feature space
        encoded = self.encoder(observation)
    
        # Ensure proper dimensions
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
              
        return encoded
        
    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create replay buffer optimized for pixel storage"""
        # Store full observation shape including frame stack
        if self.frame_stack > 1:
            buffer_obs_shape = (self.frame_stack, *self.obs_shape)
        else:
            buffer_obs_shape = self.obs_shape
        use_psr = self.training_config.prioritized_seq_replay
        if use_psr:
            return PrioritizedSequenceReplayBuffer(
                capacity=self.training_config.buffer_size,
                obs_shape=buffer_obs_shape,      # you already compute this
                action_dim=self.action_dim,
                device=self.device,
                sequence_length=getattr(self.config, "sequence_length", 10),
                overlap=getattr(self.config, "sequence_overlap", 5),
                alpha=self.training_config.alpha, 
                beta_start=self.training_config.beta0, 
                beta_end=self.training_config.beta1, 
                beta_frames=self.training_config.beta_frms, 
                eps=self.training_config.eps,
            )
        else:
            return SequenceReplayBuffer(
                capacity=self.training_config.buffer_size,
                obs_shape=buffer_obs_shape,
                action_dim=self.action_dim,
                device=self.device,
                sequence_length=self.config.sequence_length if hasattr(self.config, 'sequence_length') else 10,
                overlap=5
            )
        
    def _process_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Convert pixel observation to tensor with proper formatting"""
        if isinstance(observation, torch.Tensor):
            obs_array = observation.detach().cpu().numpy()
        else:
            obs_array = np.array(observation)
    
        # Determine observation format and convert to channels-first
        if obs_array.ndim == 2:
            # Single grayscale image (H, W) -> (1, H, W)
            obs_array = np.expand_dims(obs_array, axis=0)
        elif obs_array.ndim == 3:
            # Check if it's (H, W, C) or (C, H, W)
            if obs_array.shape[-1] in [1, 3]:  # (H, W, C) format
                obs_array = np.transpose(obs_array, (2, 0, 1))  # -> (C, H, W)
            # else assume it's already (C, H, W)
        elif obs_array.ndim == 4:
            # Could be (frame_stack, H, W, C) or (B, C, H, W)
            if obs_array.shape[-1] in [1, 3]:  # (frame_stack, H, W, C)
                obs_array = np.transpose(obs_array, (0, 3, 1, 2))  # -> (frame_stack, C, H, W)
            # else assume it's already (B, C, H, W) or (frame_stack, C, H, W)
        else:
            raise ValueError(f"Unexpected observation shape: {obs_array.shape}")
    
        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_array)
    
        # Add batch dimension if needed
        if obs_tensor.dim() == 3:  # Single image (C, H, W)
            obs_tensor = obs_tensor.unsqueeze(0)  # -> (1, C, H, W)
        elif obs_tensor.dim() == 4 and obs_tensor.shape[0] == self.frame_stack:
            # Frame stack without batch dimension (frame_stack, C, H, W)
            obs_tensor = obs_tensor.unsqueeze(0)  # -> (1, frame_stack, C, H, W)
    
        # Normalize pixel values if needed
        if obs_tensor.dtype == torch.uint8 or obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor.float() / 255.0
    
        return obs_tensor  
          
    def _process_batch_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Process batch of pixel observations"""
        # Move to device
        observations = observations.to(self.device)
        
        # Normalize to [0, 1] if needed
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        if observations.dim() == 5 :
            batch_size, frame_stack, c, h, w = observations.shape
            if hasattr(self.encoder, 'expects_frame_stack') and self.encoder.expects_frame_stack:
                pass  # Keep as is
            else:
                # Flatten frame stack into channels
                observations = observations.view(batch_size, frame_stack * c, h, w)
        return observations
        
    def train_step(self) -> Dict[str, float]:
        """
        One training iteration:
        - Uniform single-step batch for ELBO, policy (EFE), and value.
        - PER-enabled sequence-dynamics block every N steps (sample_sequences),
            with IS-weighted loss and priority update (TD + per-sequence dynamics NLL).
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        import math
        metrics: Dict[str, float] = {}

        # =========================
        # 1) Uniform single-step batch
        # =========================
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Move to device / pre-process
        obs        = self._process_batch_observations(batch['observations'])
        next_obs   = self._process_batch_observations(batch['next_observations'])
        actions    = batch['actions'].to(self.device)
        rewards    = batch['rewards'].to(self.device)
        dones      = batch['dones'].to(self.device)
        frame_idx  = batch['frame_idx'].to(self.device)
        prev_acts  = batch['prev_actions'].to(self.device)
        B0         = self.config.batch_size

        # Encode observations
        encoded_obs      = self.encode_observation(obs)
        encoded_next_obs = self.encode_observation(next_obs)
        if torch.isnan(encoded_obs).any() or torch.isinf(encoded_obs).any():
            raise ValueError("Encoded observation contains NaN/Inf")

        # Belief update (no grad)
        with torch.no_grad():
            belief_now = self.active_inference.update_belief_via_diffusion(
                encoded_obs, frame_idx=frame_idx, actions=prev_acts
            )
            latents      = belief_now['latent']
            

            belief_next = self.active_inference.update_belief_via_diffusion(
                encoded_next_obs, frame_idx=frame_idx + 1, actions=actions
            )
            next_latents = belief_next['latent']

        # -------------------------
        # 1a) Diffusion ELBO (+ contrastive) — UNWEIGHTED (uniform batch)
        # -------------------------
        self.score_optimizer.zero_grad()
        
         # Unfreeze prior during policy/value updates

        hidden_states = self.active_inference.reset_dynamics_hidden(B0)
        hidden_states = self.active_inference._reset_done_hidden(hidden_states, dones)

        elbo_loss, elbo_info = self.active_inference.compute_diffusion_elbo(
            observations=encoded_obs,
            next_observations=encoded_next_obs,
            actions=actions,
            latents=latents,
            next_latents=next_latents,
            raw_observations=obs,
            next_raw_observations=next_obs,
            frame_index=frame_idx,
            previous_actions=prev_acts,
            done_mask=dones,
            hidden_state=hidden_states
        )


        elbo_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.active_inference.latent_score_network.parameters()) +
            list(self.active_inference.latent_diffusion.parameters()) +
            list(self.encoder.parameters()) +
            list(self.active_inference.dvkl.T.parameters()) +
            list(self.active_inference.feature_decoder.parameters()) +
            (list(self.active_inference.observation_decoder.parameters())
            if isinstance(self.active_inference.observation_decoder, torch.nn.Module) else []),
            self.config.gradient_clip
        )
        self.score_optimizer.step()
        self.score_ema.update()
        metrics.update(_normalize_metrics(elbo_info))

        metrics['total_loss']       = float(elbo_loss.detach())
        # --- Train DVKL critic T and fit the Grassmann prior ---

        # -------------------------
        # 1b) Policy (EFE) — UNWEIGHTED (uniform batch)
        # -------------------------
        self.policy_optimizer.zero_grad()
        # Freeze all except policy network
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.active_inference.parameters(): p.requires_grad_(False)
        for p in self.active_inference.policy_network.parameters(): p.requires_grad_(True)

        efe, efe_info = self.active_inference.compute_expected_free_energy_diffusion(
            latents, horizon=self.config.efe_horizon
        )
        policy_loss = efe.mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_inference.policy_network.parameters(), self.config.gradient_clip)
        self.policy_optimizer.step()
        for p in self.encoder.parameters(): p.requires_grad_(True)
        for p in self.active_inference.parameters(): p.requires_grad_(True)
        metrics['policy_loss'] = float(policy_loss.detach())
        metrics.update(_normalize_metrics(efe_info, 'efe_'))
        # -------------------------
        # Train reward predictor
        # -------------------------
        self.reward_optimizer.zero_grad(set_to_none=True)
        _, _, reward_dist = self.active_inference.predict_reward_from_latent(latents.detach())
        reward_loss = -reward_dist.log_prob(rewards).mean()  # Negative log-likelihood
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_inference.reward_predictor.parameters(), self.config.gradient_clip)
        self.reward_optimizer.step()
        metrics['reward_loss'] = float(reward_loss.detach())
        # -------------------------
        # 1c) Value (critic) with EMA bootstrap — UNWEIGHTED (uniform batch)
        # -------------------------
        self.value_optimizer.zero_grad()
        logits = self.active_inference.value_network(latents)

        with torch.no_grad():
            self.value_ema.apply_shadow()
            next_logits = self.active_inference.value_network(next_latents)
            self.value_ema.restore()
            next_values = self.active_inference.value_network.expected_value(next_logits)

            lambda_returns = self.active_inference.compute_lambda_returns(
                rewards=rewards, next_values=next_values, dones=dones, lambda_=0.95, n_steps=5
            )

        value_loss = self.active_inference.value_network.loss_from_returns(logits, lambda_returns).mean()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_inference.value_network.parameters(), self.config.gradient_clip)
        self.value_optimizer.step()
        self.value_ema.update()

        metrics['value_loss'] = float(value_loss.detach())

        # (optional) epistemic estimator every few steps
        if self.total_steps % 5 == 0:
            epistemic_mi, epistemic_metrics = self.active_inference.train_epistemic_estimator(
                latents, actions, hidden_state=hidden_states, done_mask=dones
            )
            metrics['epistemic_mi'] = float(epistemic_mi)
            metrics.update(_normalize_metrics(epistemic_metrics, 'ep_'))

        # ===========================================
        # 2) PER-enabled sequence dynamics every N steps
        # ===========================================
        if (self.total_steps % 5 == 0) and hasattr(self.replay_buffer, 'sample_sequences'):
            seq_batch = self.replay_buffer.sample_sequences(self.config.batch_size // 2)
            if seq_batch is not None:
                use_per  = ('is_weights' in seq_batch) and ('tree_indices' in seq_batch)
                is_w     = seq_batch['is_weights']   if use_per else None   # [B]
                tree_idx = seq_batch['tree_indices'] if use_per else None   # [B]

                seq_obs  = seq_batch['observations']                       # [B, T, ...]
                B, T     = seq_obs.shape[:2]

                # ---- 2a) Build latent_sequences over the window (no grad) ----
                latent_sequences = []
                for t in range(T):
                    obs_t = self._process_batch_observations(seq_obs[:, t])
                    enc_t = self.encode_observation(obs_t)
                    with torch.no_grad():
                        if t == 0:
                            prev_t = (seq_batch['prev_actions'][:, 0].to(self.device)
                                    if 'prev_actions' in seq_batch else
                                    torch.zeros(B, self.action_dim, device=self.device))
                        else:
                            prev_t = seq_batch['prev_actions'][:, t - 1].to(self.device)

                        fidx_all = seq_batch.get('frame_indices', seq_batch.get('frame_idx'))
                        fidx_t = fidx_all[:, t].to(self.device) if fidx_all is not None else None

                        belief_t = self.active_inference.update_belief_via_diffusion(
                            enc_t, frame_idx=fidx_t, actions=prev_t
                        )
                    latent_sequences.append(belief_t['latent'])
                latent_sequences = torch.stack(latent_sequences, dim=1)  # [B, T, D]

                # ---- 2b) Train dynamics on sequences (IS-weighted if PER) ----
                self.dynamics_optimizer.zero_grad()
                dyn_kwargs = {}
                if is_w is not None:
                    dyn_kwargs['sample_weights'] = is_w  # importance sampling correction

                dynamics_metrics = self.active_inference.train_dynamics_on_sequence(
                    latent_sequences,                 # [B, T, D]
                    seq_batch['actions'],            # [B, T-1, A]
                    seq_batch['dones'],              # [B, T-1]
                    seq_batch['lengths'],            # [B]
                    **dyn_kwargs
                )

                dyn_loss_tensor = dynamics_metrics.get('dynamics_loss', None)
                if not torch.is_tensor(dyn_loss_tensor):
                    # safety: ensure we backward a tensor
                    dyn_loss_tensor = self.active_inference.train_dynamics_on_sequence(
                        latent_sequences,
                        seq_batch['actions'],
                        seq_batch['dones'],
                        seq_batch['lengths'],
                        **dyn_kwargs
                    )['dynamics_loss']

                dyn_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.active_inference.latent_dynamics.parameters(), self.config.gradient_clip)
                self.dynamics_optimizer.step()

                # ---- 2c) Compute priorities & update tree (PER only) ----
                if use_per and hasattr(self.replay_buffer, 'update_priorities'):
                    with torch.no_grad():
                        # TD errors per step over the window
                        D = latent_sequences.shape[-1]
                        v_all = self.active_inference.value_network.expected_value(
                            self.active_inference.value_network(latent_sequences.reshape(-1, D))
                        ).view(B, T)  # [B, T]

                        r = seq_batch['rewards'][:, :T-1].to(self.device)         # [B, T-1]
                        d = seq_batch['dones'][:, :T-1].float().to(self.device)   # [B, T-1]
                        gamma = float(self.config.discount_factor)
                        td = r + gamma * (1.0 - d) * v_all[:, 1:] - v_all[:, :-1] # [B, T-1]
                   
                        # Per-sequence dynamics NLL (masked mean)
                        t_idx = torch.arange(T-1, device=self.device).unsqueeze(0).expand(B, -1)
                        mask = (t_idx < (seq_batch['lengths'].to(self.device).unsqueeze(1) - 1)).float()  # [B, T-1]
                        td = td * mask  # Mask invalid steps
                        per_seq_sum = torch.zeros(B, device=self.device)
                        per_seq_cnt = torch.zeros(B, device=self.device)
                        log_2pi = math.log(2.0 * math.pi)
                        for t in range(T - 1):
                            cur = latent_sequences[:, t]                    # [B, D]
                            nxt = latent_sequences[:, t + 1]                # [B, D]
                            act = seq_batch['actions'][:, t].to(self.device) # [B, A]

                            next_latents_dist, _ = self.active_inference.predict_next_latent(cur, act, None)
                            step_nll = -next_latents_dist.log_prob(nxt)

                            m = mask[:, t]
                            per_seq_sum += step_nll * m
                            per_seq_cnt += m

                        eps = torch.finfo(torch.float32).eps
                        dyn_nll_seq = per_seq_sum / (per_seq_cnt + eps)  # [B]

                        # Raw priorities from TD (timewise) + per-sequence dynamics surprise
                        priorities = self.replay_buffer.compute_sequence_priority(
                            td_errors=td,                 # [B, T-1] (reduced inside as max/abs)
                            elbo_losses=None,             # add when you have per-seq ELBO from the SAME seq_batch
                            dynamics_losses=dyn_nll_seq,  # [B] (reduced inside as mean)
                            weights=(1.0, 0.0, 0.5)
                        )

                    self.replay_buffer.update_priorities(tree_idx, priorities)

                # Log sequence metrics
                metrics.update(_normalize_metrics(dynamics_metrics, 'seq_'))

        self.total_steps += 1
        return metrics


    def _setup_optimizers(self):
        """Setup optimizers including visual components"""
        # Score network optimizer (includes encoder)
        decoder_params = []
        if isinstance(self.active_inference.observation_decoder, nn.ModuleList):
            for module in self.active_inference.observation_decoder:
                decoder_params.extend(list(module.parameters()))
        else:
            decoder_params = list(self.active_inference.observation_decoder.parameters())

        model_params = list(self.active_inference.latent_score_network.parameters()) + \
                       list(self.active_inference.latent_diffusion.parameters()) + \
                       list(self.encoder.parameters()) + \
                       list(self.active_inference.feature_decoder.parameters()) + \
                       decoder_params
        dv_params    = list(self.active_inference.dvkl.T.parameters())
        self.score_optimizer = torch.optim.AdamW([
            {"params": model_params, "lr": self.config.learning_rate, "weight_decay": 1e-5},
            {"params": dv_params,    "lr": self.config.dv_lr,         "weight_decay": 0.0},
        ], betas=(0.9, 0.999))

        #reward predictor optimizer
        self.reward_optimizer = torch.optim.AdamW(
            self.active_inference.reward_predictor.parameters(),
            lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=1e-6
        )

        # Policy optimizer
        self.policy_optimizer = torch.optim.AdamW(
            self.active_inference.policy_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Value optimizer
        self.value_optimizer = torch.optim.AdamW(
            self.active_inference.value_network.parameters(),
            lr=self.config.learning_rate
        )
    
        
        # Dynamics optimizer
        self.dynamics_optimizer = torch.optim.AdamW(
            self.active_inference.latent_dynamics.parameters(),
            lr=self.config.learning_rate
        )
        #Add epistemic optimizer
        self.epistemic_optimizer = torch.optim.AdamW(
            self.active_inference.epistemic_estimator.parameters(),
            lr=self.config.learning_rate*0.1,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        self.active_inference.epistemic_optimizer = self.epistemic_optimizer


