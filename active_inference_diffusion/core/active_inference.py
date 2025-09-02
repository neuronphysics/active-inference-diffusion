"""
Active Inference with Diffusion-Generated Latent Spaces
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.func import linearize
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from ..configs import ActiveInferenceConfig
from ..encoder.visual_encoders import ConvDecoder
from .diffusion import LatentDiffusionProcess
from ..models.score_networks import LatentScoreNetwork
from ..models.policy_networks import DiffusionConditionedPolicy
from ..models.value_networks import ValueNetwork
from ..models.dynamics_models import LatentDynamicsModel, TransformerDynamicsModel
from .free_energy import FreeEnergyComputation
from .divergence import DVKL
from ..utils.util import symexp, symlog, DiscDist
HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor], None]

class DiffusionActiveInference(nn.Module):
    """
    Core Active Inference implementation with diffusion-generated latents

    Key innovations:
    - Latent beliefs emerge from reverse diffusion process
    - Policies conditioned on continuous latent manifolds
    - Expected Free Energy computed over diffusion trajectories
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        latent_dim: int,
        config: "ActiveInferenceConfig",
        pixel_shape: Optional[Tuple[int, int, int]] = None,
        shared_visual_encoder: Optional[nn.Module] = None,
        checkpoint_decoder: bool = True,
        checkpoint_reward_predictor: bool = True,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.config = config
        self.pixel_shape = pixel_shape
        self.is_pixel_observation = config.pixel_observation
        self.epistemic_dropout_rate = 0.2
        self.device = torch.device(config.device)
        self.shared_visual_encoder = shared_visual_encoder
        if self.is_pixel_observation and pixel_shape is not None:
            self.raw_observation_shape = pixel_shape
        else:
            self.raw_observation_shape = None
        self.ckpt_decoder = checkpoint_decoder
        self.ckpt_reward_predictor = checkpoint_reward_predictor
        # Initialize components
        self._build_models()
        self.to(self.device)

        # Current belief state (diffusion-generated)
        self.current_latent = None
        self.latent_trajectory = []

    def _build_models(self):
        """Build core models for diffusion active inference"""

        # Latent diffusion process
        self.latent_diffusion = LatentDiffusionProcess(
            self.config.diffusion, latent_dim=self.latent_dim
        )
        # Add reward preference components
        self.register_buffer("reward_mean", torch.tensor(0.0).to(self.device))
        self.register_buffer("reward_var", torch.tensor(1.0).to(self.device))
        self.register_buffer(
            "preference_temperature",
            torch.tensor(self.config.preference_temperature).to(self.device),
        )

        # Score network for latent generation

        self.latent_score_network = LatentScoreNetwork(
            latent_dim=self.latent_dim,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            use_attention=True,
        )

        # Policy network conditioned on diffusion latents

        self.policy_network = DiffusionConditionedPolicy(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.config.hidden_dim,
            squash_output=True
        )

        # Value network for latent states

        self.value_network = ValueNetwork(
            state_dim=self.latent_dim,  # Using latent dimension as state dimension
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.value_net_num_layers,
            num_bins=self.config.num_value_bins,
        )
        

        # Dynamics model in latent space
        if self.config.dynamics_type == "transformer":
            self.latent_dynamics = TransformerDynamicsModel(
                state_dim=self.latent_dim,
                action_dim=self.action_dim,
                K=self.config.grassmann_dynamics_K,
                r=self.config.grassmann_dynamics_r,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.dynamics_num_layers,
                n_heads=self.config.dynamics_n_heads,
                dropout=self.config.dynamics_dropout,
                context_len=self.config.dynamics_context_len,
                attn_impl=self.config.dynamics_attn_impl,   # {"auto","flash","mem","math"} for SDPA backends
            )
        else:
            self.latent_dynamics = LatentDynamicsModel(
                state_dim=self.latent_dim,
                action_dim=self.action_dim,
                hidden_dim=self.config.hidden_dim,
                K=self.config.grassmann_dynamics_K,
                r=self.config.grassmann_dynamics_r,
                lstm_hidden=self.config.hidden_dim//2,
                trunk_hidden=self.config.hidden_dim,
                trunk_layers=self.config.dynamics_num_layers,
                dropout=self.config.dynamics_dropout,
            )
        self.current_hidden_state = None  # Initialize hidden state for dynamics
        # Observation decoder (latent -> observation prediction)
        if not self.is_pixel_observation:
            self.observation_decoder = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.latent_dim, self.config.hidden_dim * 2),
                        nn.LayerNorm(self.config.hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(self.epistemic_dropout_rate),
                    ),
                    nn.Sequential(
                        nn.Linear(
                            self.config.hidden_dim * 2, self.config.hidden_dim * 2
                        ),
                        nn.LayerNorm(self.config.hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(self.epistemic_dropout_rate),
                    ),
                    nn.Sequential(
                        nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
                        nn.LayerNorm(self.config.hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(self.epistemic_dropout_rate),
                    ),
                    nn.Linear(self.config.hidden_dim, self.observation_dim),
                ]
            )
            observation_shape = self.observation_dim  # For non-pixel observations
        else:
            self.observation_decoder = ConvDecoder(
                latent_dim=self.latent_dim,
                output_dim=np.prod(self.pixel_shape),  # Total pixel count
                img_channels=self.pixel_shape[0],
                hidden_dim=self.config.hidden_dim,
                spatial_size=21,
                frame_stack=self.config.frame_stack,
            )
            # Also add a feature decoder for reconstructing encoded features
            self.feature_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.epistemic_dropout_rate),
                nn.Linear(
                    self.config.hidden_dim, self.latent_dim
                ),  # Decode to feature space
                nn.Tanh(),
            )
            c, h, w = self.pixel_shape
            stacked_shape = (c * self.config.frame_stack, h, w)
            observation_shape = stacked_shape  # For pixel observations
        self.free_energy = FreeEnergyComputation(
            precision_init=1.0,
            observation_decoder=self.observation_decoder,
            is_pixel_observation=self.is_pixel_observation,
        ).to(self.device)

        # Epistemic estimator for latent uncertainty
        self.epistemic_estimator = FunctionSpaceEpistemicEstimator(
            decoder=self.observation_decoder,
            feature_extractor=self.shared_visual_encoder,
            latent_dim=self.latent_dim,
            observation_shape=observation_shape,
            is_pixel=self.is_pixel_observation,
            device=self.device,
            hidden_dim=self.config.hidden_dim,
            jac_dim=self.config.spatial_aggregator_output_dim,
            latent_proj_dim=self.latent_dim,   
            use_checkpointing=True,     # Enable checkpointing
            checkpoint_critic=True,
            checkpoint_jacproj=True,
            checkpoint_latproj=True,
            robust_marginals=False,     
        )
        self.num_reward_bins = getattr(self.config, "num_reward_bins", 255)
        #initialize DVKL for divergence between posterior and prior 
        
        self.dvkl_T = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dim), 
            nn.LayerNorm(self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim), 
            nn.LayerNorm(self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, 1)
          )

        self.dvkl = DVKL(self.dvkl_T, ema_momentum=self.config.dv_ema)

        # Initialize a reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, self.num_reward_bins),
        )
        nn.init.zeros_(self.reward_predictor[-1].weight)
        nn.init.zeros_(self.reward_predictor[-1].bias)
        # Initialize final layer weights to small values
        nn.init.kaiming_normal_(
            self.reward_predictor[-1].weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.zeros_(self.reward_predictor[-1].bias)

    def to(self, device):
        """Override to ensure ALL components move to device"""
        # Convert device to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)

        # Call parent to() method
        super().to(device)

        # Update our device attribute
        self.device = device

        # Explicitly move all components
        self.latent_diffusion = self.latent_diffusion.to(device)
        self.latent_score_network = self.latent_score_network.to(device)
        self.policy_network = self.policy_network.to(device)
        self.value_network = self.value_network.to(device)
        self.latent_dynamics = self.latent_dynamics.to(device)
        self.reward_predictor = self.reward_predictor.to(device)

        # Handle observation decoder based on type
        if isinstance(self.observation_decoder, nn.ModuleList):
            # For state observations - move each module in the list
            for i in range(len(self.observation_decoder)):
                self.observation_decoder[i] = self.observation_decoder[i].to(device)
        else:
            # For pixel observations
            self.observation_decoder = self.observation_decoder.to(device)

        # Move feature decoder if it exists
        if hasattr(self, "feature_decoder"):
            self.feature_decoder = self.feature_decoder.to(device)

        # Move epistemic estimator with explicit device update
        self.epistemic_estimator = self.epistemic_estimator.to(device)
        self.epistemic_estimator.device = device  # Update its device attribute

        # Move buffers
        self.reward_mean = self.reward_mean.to(device)
        self.reward_var = self.reward_var.to(device)
        self.preference_temperature = self.preference_temperature.to(device)

        return self

    def reset_dynamics_hidden(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset LSTM hidden state for new trajectories"""
        return self.latent_dynamics.init_hidden(batch_size, self.device)

    def _clone_dyn_hidden(self, h: HiddenState) -> HiddenState:
        if h is None: 
            return None
        if isinstance(h, tuple):
            # LSTM
            return (h[0].clone(), h[1].clone())
        if isinstance(h, dict):
            # Transformer
            return {k: v.clone() for k, v in h.items()}
        return h

    def _reset_done_hidden(self, h: HiddenState, done: torch.Tensor) -> HiddenState:
        """Reset only those envs where done==1, preserving others."""
        if h is None or done is None:
            return h
        if isinstance(h, tuple):
            h0, c0 = h
            dm = done.float().view(1, -1, 1)  # [1,B,1] to broadcast over LSTM layers
            # Zero the done envs; keep others
            h0 = h0 * (1.0 - dm)
            c0 = c0 * (1.0 - dm)
            return (h0, c0)
        if isinstance(h, dict):
            out = {k: v for k, v in h.items()}
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                if "tokens" in out: out["tokens"][idx].zero_()
                if "lengths" in out: out["lengths"][idx] = 0
            return out
        return h


    def _blend_hidden(self, new: HiddenState, done: torch.Tensor) -> HiddenState:
        """
        Keep `new` where not done, insert a freshly reset hidden where done.
        (Useful if you want to combine step output with per-env resets.)
        """
        if new is None:
            return None

        # LSTM hidden: tuple(h, c) with shapes [L, B, H]
        if isinstance(new, tuple):
            h, c = new
            # mask shape: [1, B, 1] to broadcast across layers & hidden dim
            dm = done.to(h.dtype).view(1, -1, 1)
            # get a fresh reset state for this batch size
            reset_h, reset_c = self.reset_dynamics_hidden(h.size(1))
            # blend: reset where done=1, keep new where done=0
            h = reset_h * dm + h * (1.0 - dm)
            c = reset_c * dm + c * (1.0 - dm)
            return (h, c)

        # Dict-style hidden (e.g., token caches, lengths, etc.)
        if isinstance(new, dict):
            out = {k: v.clone() for k, v in new.items()}  # avoid in-place on shared graph
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                reset = self.reset_dynamics_hidden(len(done))
                # copy only for keys that exist in both; leave others untouched
                if isinstance(reset, dict):
                    for k in out.keys():
                        if k in reset:
                            out[k][idx] = reset[k][idx]
                else:
                    # fallback: zero the done indices if reset isn't a dict
                    for k in out.keys():
                        out[k][idx] = torch.zeros_like(out[k][idx])
            return out

        # Fallback for other hidden types (e.g., GRU tensor) — untouched
        return new

    def decode_observation(
        self, latent: torch.Tensor, decode_to_pixels: bool = True
    ) -> torch.Tensor:
        """
        decoding that can decode to either pixels or features

        Args:
            latent: Latent representation
            decode_to_pixels: If True and using pixel observations, decode to raw pixels.
                            If False, decode to encoded feature space.
        """
        latent = latent.to(self.device)
        use_checkpointing = self.ckpt_decoder and latent.requires_grad
        if self.is_pixel_observation:
            if decode_to_pixels:
                # Decode to pixel space
                if use_checkpointing and self.training:
                    return cp.checkpoint(self.observation_decoder, latent, use_reentrant=False)
                else:
                    return self.observation_decoder(latent)
            else:
                # Decode to feature space (for reconstruction loss)
                self.feature_decoder = self.feature_decoder.to(self.device)
                if use_checkpointing and self.training:
                    return cp.checkpoint(self.feature_decoder, latent, use_reentrant=False)
                else:
                    return self.feature_decoder(latent)
        else:
            # For non-pixel observations, use fully connected decoder
            h = latent
            if use_checkpointing and self.training:
                h1 = cp.checkpoint(self.observation_decoder[0], h, use_reentrant=False)
                h2 = cp.checkpoint(self.observation_decoder[1], h1, use_reentrant=False)
                h2 = h2 + h1  # Skip connection
                h3 = cp.checkpoint(self.observation_decoder[2], h2, use_reentrant=False)
            else:
                h1 = self.observation_decoder[0](h)
                h2 = self.observation_decoder[1](h1)
                h2 = h2 + h1  # Skip connection
                h3 = self.observation_decoder[2](h2)
            return self.observation_decoder[3](h3)

    def predict_reward_from_latent(
        self, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use existing reward predictor to get reward distribution from latent
        """
        latent = latent.to(self.device)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            raise ValueError("Latent tensor contains NaN or Inf values")
        if self.ckpt_reward_predictor and latent.requires_grad and self.training:
            logits = cp.checkpoint(self.reward_predictor, latent, use_reentrant=False)
        else:
            logits = self.reward_predictor(latent)
        dist = DiscDist(logits, low=self.config.reward_disc_low, high=self.config.reward_disc_high, device=logits.device)
        pred = dist.mean().squeeze(-1)
        return pred, logits, dist

    def update_belief_via_diffusion(
        self,
        observation: torch.Tensor,
        raw_observation: Optional[torch.Tensor] = None,
        num_trajectories: int = 5,
        frame_idx: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Update belief using reverse diffusion process
        This is the core innovation - beliefs as diffusion-generated latents
        Returns different statistics based on batch size:
        - batch_size=1: Returns uncertainty estimates for single observation
        - batch_size>1: Returns per-sample latents without population statistics

        """
        observation = observation.to(self.device)
        if frame_idx is not None:
            frame_idx = frame_idx.to(self.device)
        if actions is not None:
            actions = actions.to(self.device)
        # Handle different input shapes
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            batch_size = 1
        else:
            batch_size = observation.shape[0]
        with torch.no_grad():
            # Save training states of all components involved in belief generation
            training_states = {
                "score_network": self.latent_score_network.training,
                "diffusion": self.latent_diffusion.training,
            }

            # Set to eval mode for deterministic belief generation
            self.latent_score_network.eval()
            self.latent_diffusion.eval()

            if batch_size == 1:
                # Expand observation to run multiple trajectories in parallel
                expanded_obs = observation.expand(num_trajectories, -1)

                with torch.autograd.set_detect_anomaly(True):
                    # Generate multiple trajectories in parallel
                    trajectories = self.latent_diffusion.generate_latent_trajectory(
                        score_network=self.latent_score_network,
                        batch_size=num_trajectories,  # Run num_trajectories in parallel
                        observation=expanded_obs,
                        deterministic=False,
                        frame_time= frame_idx,
                        action= actions
                    )


                self.current_latent = trajectories[-1]  # Keep all trajectories for uncertainty estimation

                # Store the full trajectory for analysis
                self.latent_trajectory = trajectories
                trajectory_length = len(trajectories)
            else:
                # Generate latent via reverse diffusion conditioned on observation
                with torch.autograd.set_detect_anomaly(True):
                    trajectories = self.latent_diffusion.generate_latent_trajectory(
                        score_network=self.latent_score_network,
                        batch_size=batch_size,
                        observation=observation,
                        deterministic=False,
                        frame_time= frame_idx,
                        action = actions
                    )

                # Final latent is the belief
                self.current_latent = trajectories[-1]
                self.latent_trajectory = trajectories

                trajectory_length = len(trajectories)
        self.latent_score_network.train(training_states["score_network"])
        self.latent_diffusion.train(training_states["diffusion"])

        # Validate outputs before returning
        if (
            torch.isnan(self.current_latent).any()
            or torch.isinf(self.current_latent).any()
        ):
            raise ValueError(
                "Generated latents contain NaN or Inf values inside belief update!"
            )

        return {
            "latent": self.current_latent,
            "trajectory_length": trajectory_length,
            "observation": observation,
            "raw_observation": raw_observation,
        }

    def compute_expected_free_energy_diffusion(
        self,
        latent: torch.Tensor,
        horizon: int = 5,
        num_trajectories: int = 6,
        num_ambiguity_samples: int = 2,
        hidden_state: HiddenState= None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expected Free Energy over diffusion-generated trajectories
        G(π) = Epistemic + Pragmatic + Consistency terms
        G(π) = E_z~p_θ(z) E_q(s'|s,π) [D_KL[q(o'|s')||p(o')] - log p_φ(π|z)]
        """
        latent = latent.to(self.device)
        device = latent.device
        batch_size = latent.shape[0]
        if hidden_state is None:
            hidden_state = self.reset_dynamics_hidden(batch_size)
        # Initialize accumulators
        total_efe = torch.zeros(batch_size, device=device)
        epistemic_values = []
        pragmatic_values = []
        latent_consistency = []

        # Generate future latent trajectories
        for traj_idx in range(num_trajectories):
            current_latent = latent.clone()
            traj_efe = 0
            hidden_state = self._clone_dyn_hidden(hidden_state)  # Reset hidden state for each trajectory
            dm = done_mask
            for t in range(horizon):
                # Sample policy from current latent
                
                action, log_prob, policy_dist = self.policy_network(current_latent)

                # Predict next latent
                next_latent_dist, hidden_state = (
                    self.predict_next_latent(current_latent, action, hidden_state, done_mask=dm)
                )
                dm = None  # Only apply done mask at first step
                # 1. Pragmatic value (reward prediction)
                # For p(o) ∝ exp(r(o)/τ), we have ln p(o) = r(o)/τ - ln Z
                # Since Z is constant across policies, we can ignore it
                next_latent, _ = next_latent_dist.sample(B=batch_size, temperature=1.0, hard=True, device=device)
                predicted_reward, _, _ = self.predict_reward_from_latent(next_latent)
                # This makes high-reward states preferred under EFE p(o) ∝ exp(r(o)/τ)
                pragmatic = predicted_reward / self.preference_temperature

                # Pragmatic value: Expected value under policy
                
                value = self.value_network(next_latent)                       # [B, K]
                exceptional_value = self.value_network.expected_value(value)  # [B], ∑ p(bin)*bin_real
                pragmatic += exceptional_value

                # 2. Consistency (negative policy entropy)-> exploration bonus
                consistency = -self.policy_network.get_policy_entropy(current_latent)

                epistemic, epistemic_metrics = self.compute_epistemic_value(
                    next_latent,
                    num_samples=num_ambiguity_samples,
                )

                # Accumulate EFE
                step_efe = (
                    self.config.epistemic_weight * epistemic
                    + self.config.pragmatic_weight * pragmatic
                    + self.config.consistency_weight * consistency
                )

                traj_efe += (self.config.discount_factor**t) * step_efe

                # Update for next step
                current_latent = next_latent

            total_efe += traj_efe / num_trajectories

            # Store components for analysis
            epistemic_values.append(epistemic)
            pragmatic_values.append(pragmatic)
            latent_consistency.append(consistency)

        info = {
            "epistemic_mean": torch.stack(epistemic_values).mean(),
            "pragmatic_mean": torch.stack(pragmatic_values).mean(),
            "consistency_mean": torch.stack(latent_consistency).mean(),
            "num_trajectories": num_trajectories,
            "horizon": horizon,
            **epistemic_metrics,
        }

        return total_efe, info

    def compute_epistemic_value(
        self,
        next_latent: torch.Tensor,
        num_samples: int = 4,
    ) -> torch.Tensor:
        # Compute epistemic value: H(o|s,π) - H(o|s,θ,π)
        # Epistemic value (ambiguity - observation uncertainty)
        # - H[p(o|s,π)] is entropy marginalizing over model parameters (using dropout)
        # - H[p(o|s,θ,π)] is entropy for a fixed set of parameters
        next_latent = next_latent.to(self.device)
        with torch.no_grad():
            epistemic_value, metrics = self.epistemic_estimator(
                next_latent, num_samples
            )

        return epistemic_value, metrics

    def train_epistemic_estimator(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Train MINE estimator separately"""
        latents = latents.to(self.device)
        actions = actions.to(self.device)
        done_mask = done_mask.to(self.device) if done_mask is not None else None
        # Predict next latent distribution
        next_latent_dist, _ = self.predict_next_latent(latents, actions, hidden_state, done_mask)
        next_latents, _ = next_latent_dist.sample(B=latents.shape[0], hard=False, device=self.device)

        # Compute MINE loss (negative MI for minimization)
        mi_estimate, metrics = self.epistemic_estimator(next_latents)
        total_loss = -mi_estimate.mean()

        # Optimize
        self.epistemic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.epistemic_estimator.parameters(), self.config.gradient_clip
        )
        self.epistemic_optimizer.step()

        return mi_estimate.mean().item(), metrics


    def predict_next_latent(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden_state: HiddenState = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, HiddenState]:
        """Predict next latent state using learned dynamics"""
        latent = latent.to(self.device)
        action = action.to(self.device)
        prior, new_hidden_state = self.latent_dynamics(latent, action, hidden_state, done_mask)

        return prior, new_hidden_state


    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        raw_observation: Optional[torch.Tensor] = None,
        maintain_hidden_state: bool = True,
        from_idx: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action using diffusion-generated latent active inference
        """
        observation = observation.to(self.device)
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # Update belief via diffusion
        belief_info = self.update_belief_via_diffusion(observation, raw_observation, frame_idx=from_idx, actions=actions)
        batch_size = observation.shape[0]

        if not maintain_hidden_state or self.current_hidden_state is None:
            # Reset hidden state if not maintaining or not set
            self.current_hidden_state = self.reset_dynamics_hidden(batch_size)
        # Current latent belief
        latent = belief_info["latent"]
        # Ensure latent has proper shape for policy
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        # Compute expected free energy
        efe, efe_info = self.compute_expected_free_energy_diffusion(
            latent,
            horizon=self.config.efe_horizon,
            hidden_state=self.current_hidden_state,
            done_mask=None,
        )

        # Get action from policy conditioned on latent
        action, log_prob, policy_dist = self.policy_network(
            latent, deterministic=deterministic
        )
        _, self.current_hidden_state = self.predict_next_latent(
            latent, action, self.current_hidden_state
        )
        action = action.cpu()
        # Ensure action has proper shape
        if action.dim() > 2:
            action = action.squeeze()
        elif action.dim() == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        elif action.dim() == 0:
            # Handle scalar actions
            action = action.unsqueeze(0)

        # Compile information
        info = {
            **belief_info,
            "expected_free_energy": efe.mean().cpu().item(),
            "action_log_prob": log_prob.mean().cpu().item(),
            "policy_entropy": self.policy_network.get_policy_entropy(latent).mean().cpu().item(),
            **{
                k: v.cpu().item() if torch.is_tensor(v) else v
                for k, v in efe_info.items()
            },
        }

        return action, info

    def compute_diffusion_elbo(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        raw_observations: Optional[torch.Tensor] = None,
        next_raw_observations: Optional[torch.Tensor] = None,
        frame_index: Optional[torch.Tensor] = None,
        previous_actions: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[HiddenState] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Annealing time conditioned ELBO for diffusion-generated latents
        ELBO for diffusion-generated latent active inference

        L = E_q(z|o,π)[log p(o|z)] - D_KL[q(z|o,π)||p_θ(z)] + R_diffusion(θ)
        accuracy − complexity (+ diffusion term)
        - accuracy: reconstruction log-likelihood proxy (MSE)
        - complexity: KL(q||p) approximated by −E_q[log p_G(z)] using the Grassmann prior
        - diffusion: score matching term for the latent diffusion

        """
        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        actions = actions.to(self.device)
        batch_size = observations.shape[0]
        device = self.device

        # Generate latents if not provided
        if latents is None:
            # Use current belief generation
            belief_info = self.update_belief_via_diffusion(
                          observations, 
                          raw_observations, 
                          frame_idx=frame_index, 
                          actions=previous_actions
            )
            latents = belief_info["latent"]

            if (
                torch.isnan(latents).any()
                or torch.isinf(latents).any()
            ):
                raise ValueError(
                    "Latent mean, std, or latent contains NaN or Inf values during belief update"
                )
        else:
            # Use provided latents
 
            latents = latents
        if next_latents is None:
            next_belief_info = self.update_belief_via_diffusion(
                next_observations, next_raw_observations,
                frame_idx=frame_index + 1 if frame_index is not None else None,
                actions=actions
            )
            next_latents = next_belief_info['latent']
        else:
            next_latents = next_latents
        if hidden_state is None:
            hidden_state = self.latent_dynamics.init_hidden(batch_size, self.device)
        
        # Reset hidden state for done episodes
        if done_mask is not None:
            hidden_state = self._reset_done_hidden(hidden_state, done_mask)
        
        # KL term with annealing and Grassmannian prior E_q[T]-E_q[e^T]
        with torch.no_grad():
            prior, hidden = self.latent_dynamics(
                state=latents.detach(),             # stop-grad into q for stability in early training
                action=actions.detach(),
                hidden=hidden_state,     # or a per-batch hidden you reset outside
                done_mask=done_mask
            )
        kl_loss = self.dvkl(next_latents, prior)
                     # H[p] ≈ -E_p[log p]

        # Reconstruction term (accuracy:-log p(o|z))
        accuracy_loss, accuracy_info = self.free_energy.compute_accuracy(
            states=next_latents,                    # Your latent states z
            observations=next_observations,         # Encoded features (for pixels) or states
            raw_observations=next_raw_observations,  # Pass raw pixels if available
        )

        
        # Diffusion score matching loss
        # Sample continuous time with importance sampling
        # Emphasize times where loss is typically high
        if hasattr(self, "time_importance_weights"):
            # Use learned importance weights
            t = self._importance_sample_time(batch_size, device)
        else:
            # Uniform sampling initially
            t = torch.rand(batch_size, device=device)

        noise = torch.randn_like(latents, device=device)

        noisy_latents, true_noise, sample_info = (
            self.latent_diffusion.continuous_q_sample(next_latents, 
                                                      t, 
                                                      noise, 
                                                      frame_time=frame_index, 
                                                      actions=actions)
        )
        # Predict score
        predicted_score = self.latent_score_network(noisy_latents, t, next_observations, frame_time=frame_index, action=actions)
        # Compute true score with proper scaling

        sigma = sample_info["sigma"]

        # True score: -noise / sigma (not sqrt(1-alpha) for continuous time)
        true_score = -noise / (sigma + torch.finfo(sigma.dtype).eps)

        # Annealed loss weight
        loss_weight = self.latent_diffusion.compute_loss_weight(t)
        # Score matching loss with annealing
        # shape: (batch_size, latent_dim)
        per_sample_losses = loss_weight.view(-1) * torch.sum((predicted_score - true_score)**2, dim=1)
        score_matching_loss = torch.mean(per_sample_losses)


        kl_weight = torch.exp(-5.0 * t.mean())  # Anneal KL over time
        
        # Total ELBO (to maximize)= accuracy - complexity + entropy - diffusion
        elbo = (
            -accuracy_info["reconstruction_mse"] #minimize positive reconstruction error (Maximize accuracy)
            - self.config.kl_weight * kl_loss * kl_weight #minimize positive complexity
            - self.config.diffusion_weight * score_matching_loss
        )
        self._update_time_importance(t, per_sample_losses.detach())
        # Update precision based on balance
        self.free_energy.update_precision(
            complexity=kl_loss.detach(),
            accuracy=accuracy_info["accuracy"]
        )
        info = {
            "reconstruction_loss": accuracy_info["reconstruction_mse"].item(),
            "kl_loss": kl_loss.item(),
            "score_matching_loss": score_matching_loss.item(),
            "elbo": elbo.item(),
            "mean_time": t.mean().item(),
            "loss_weight_mean": loss_weight.mean().item(),
            "hidden_state":hidden
        }

        return -elbo, info  # Return negative ELBO as loss


    def train_dynamics_on_sequence(
        self,
        latent_sequences: torch.Tensor,  # [batch_size, seq_len, latent_dim]
        action_sequences: torch.Tensor,  # [batch_size, seq_len-1, action_dim]
        done_sequences: torch.Tensor,  # [batch_size, seq_len-1]
        sequence_lengths: torch.Tensor,  # [batch_size] actual lengths
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train dynamics with proper hidden state flow and done masking
        """
        batch_size, max_seq_len = latent_sequences.shape[:2]
        device = self.device
        hidden_states = self.reset_dynamics_hidden(batch_size)
        # Initialize hidden states for all sequences
       

        total_loss = 0


        for t in range(max_seq_len - 1):
            # Create mask for valid time steps
            mask = (t < sequence_lengths - 1).float().to(device)
            if mask.sum() == 0:
                break

            z_t = latent_sequences[:, t]
            z_tp1 = latent_sequences[:, t + 1]
            a_t = action_sequences[:, t]
            d_t = done_sequences[:, t]

            # Predict next latent with hidden state
            prior, new_hidden_states = self.predict_next_latent(
                z_t, a_t, hidden_states, done_mask=d_t
            )
            z_hat, _ = prior.sample(B=z_t.size(0), temperature=1.0, hard=False, device=device)  # [B, D]

            # InfoNCE: positives on the diagonal, negatives are other items in the batch
            # Cosine similarity (normalized) with temperature τ
            z_hat_n = F.normalize(z_hat, dim=-1)
            z_tp1_n = F.normalize(z_tp1, dim=-1)
            logits  = (z_hat_n @ z_tp1_n.t()) / self.config.contrastive_tau  # [B, B]
            labels  = torch.arange(z_t.size(0), device=device)                # positives are diagonal

            # mask invalid steps (set their logits to -inf so they don't contribute)
            if mask.sum() < mask.numel():
                valid = mask.bool()
                logits = logits[valid][:, valid]
                labels = torch.arange(valid.sum(), device=device)

            step_loss = F.cross_entropy(logits, labels)

            if sample_weights is not None:
                w = (sample_weights * mask).sum() / (mask.sum() + torch.finfo(torch.float32).eps)
                total_loss = total_loss + w * step_loss
            else:
                total_loss = total_loss + step_loss

            # reset/blend hidden where episodes ended at t
            hidden_states = self._blend_hidden(new=new_hidden_states, done=d_t)

        avg_loss = total_loss / max(1, (sequence_lengths - 1).clamp_min(0).sum().item())
        return {
            "dynamics_loss": avg_loss,
            "valid_steps": int((sequence_lengths - 1).clamp_min(0).sum().item()),
        }

    def compute_lambda_returns(
        self,
        rewards: torch.Tensor,       # [B]
        next_values: torch.Tensor,   # [B], where next_values[t] = V(s_{t+1})
        dones: torch.Tensor,         # [B] in {0,1} for transition t -> t+1
        lambda_: float = 0.95,
        n_steps: int = 5,
        exclude_immediate_rewards: bool = False,
    ) -> torch.Tensor:
        """
        Correct n-step λ-returns with proper bootstrapping index:
        G_t^(λ,n) = r_t + γ (1-d_t) [ (1-λ) V_{t+1} + λ G_{t+1} ]  (unrolled n steps)
        where V_{t+1} = next_values[t], and the final bootstrap at horizon J
        uses V_{t+J} = next_values[t+J-1].

        Assumes 'batch as time' (i.e., index t+1 is the next transition in the batch).
        """
        assert rewards.device == next_values.device == dones.device
        assert rewards.dtype == next_values.dtype

        B      = rewards.shape[0]
        out    = torch.zeros_like(rewards)
        gamma  = float(self.config.discount_factor)
        lam    = float(lambda_)
        one_m  = 1.0 - lam
        dones  = dones.float()

        # scalar zero on the correct device/dtype (no grads, no 0*x)
        zero = rewards.new_zeros(())

        for t in range(B):
            # horizon length for this start
            J = min(int(n_steps), B - t)
            if J <= 0:
                out[t] = zero
                continue

            # Bootstrap V(s_{t+J}) which is stored at next_values[t+J-1]
            idx_boot = t + J - 1
            if idx_boot >= B:  # defensive bound (shouldn't fire with the J formula above)
                idx_boot = B - 1
            G = next_values[idx_boot]

            # Backward recursion: k = J-1 ... 0
            for k in range(J - 1, -1, -1):
                idx = t + k

                r = rewards[idx]
                if exclude_immediate_rewards and k == 0:
                    r = zero  # cleaner than 0.0 * rewards[idx]

                m  = 1.0 - dones[idx]      # stop bootstrapping at terminals
                v1 = next_values[idx]      # V(s_{idx+1})

                # G_k = r_k + γ m_k [ (1-λ) V_{k+1} + λ G_{k+1} ]
                G = r + gamma * m * (one_m * v1 + lam * G)

            out[t] = G

        return out


    def _importance_sample_time(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Sample time with importance weights based on loss history"""
        if not hasattr(self, "time_importance_weights"):
            # Initialize uniform
            self.time_importance_weights = torch.ones(100, device=device)

        # Sample from categorical distribution
        probs = F.softmax(self.time_importance_weights, dim=0)
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # Convert to continuous time
        t = (indices.float() + torch.rand(batch_size, device=device)) / 100.0

        return t

    def _update_time_importance(self, t: torch.Tensor, loss: torch.Tensor):
        """Update importance weights for time sampling"""
        if not hasattr(self, "time_importance_weights"):
            self.time_importance_weights = torch.ones(100, device=t.device)

        # Discretize time
        indices = (t * 99).long().clamp(0, 99)
        if loss.dim() > 1:
            # If loss has multiple dimensions, reduce to scalar per sample
            loss = loss.view(loss.shape[0], -1).sum(dim=1)

        # Update weights with EMA
        for i in range(len(indices)):
            time_bin = indices[i].item()
            sample_loss = loss[i].item()
            current_weight = self.time_importance_weights[time_bin].item()
            new_weight = 0.99 * current_weight + 0.01 * sample_loss
            self.time_importance_weights[time_bin] = new_weight


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract coefficients helper"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class EMAModel:
    """Exponential Moving Average of model weights for stable training"""

    def __init__(self, model, decay=0.9999, device=None):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply EMA weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + 1e-6) / input.shape[0]
        )
        return grad, None


def ema_loss(x, running_mean, alpha=0.01):
    """Exponential moving average loss for stable MINE training"""
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = alpha * t_exp + (1.0 - alpha) * running_mean.item()
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class FunctionSpaceEpistemicEstimator(nn.Module):
    """
    Memory-efficient epistemic estimator using JVP features of decoder outputs,
    passed through a *provided* feature_extractor (built outside, e.g., in your
    state agent or pixel agent).

    Clean API:
      - decoder: nn.Module mapping latent z -> observation x
      - feature_extractor: nn.Module mapping observation x -> feature vector
      - is_pixel: whether x is [C,H,W] (True) or flat [D] (False)
      - No internal construction of encoders. No AMP/mixed precision.

    Forward returns a per-batch epistemic score based on a MINE-DV lower bound:
       I(Z; Φ(J_z f)) ≈ E_joint[T] - log E_marg[exp T]
    where Φ encodes JVPs of f (decoder) along random directions in z-space.

    Returns a per-batch epistemic score (MINE-DV bound) and metrics.
    """

    def __init__(
        self,
        *,
        decoder: nn.Module,
        feature_extractor: nn.Module,
        latent_dim: int,
        observation_shape: Union[int, Tuple[int, int, int]],
        is_pixel: bool,
        device: Union[str, torch.device] = "cuda",
        ntk_samples: int = 3,
        jvp_chunk_size: int = 1,
        hidden_dim: int = 256,
        jac_dim: int = 128,
        latent_proj_dim: int = 128,
        use_checkpointing: bool = False,
        checkpoint_critic: Union[bool, None] = None,
        checkpoint_jacproj: Union[bool, None] = None,
        checkpoint_latproj: Union[bool, None] = None,
        robust_marginals: bool = False,
        eps: float = torch.finfo(torch.float16).eps,
    ) -> None:
        super().__init__()

        # External modules (you provide them)
        self.decoder = decoder
        self.feature_extractor = feature_extractor

        # Shapes / flags
        self.latent_dim = int(latent_dim)
        self.observation_shape = observation_shape
        self.is_pixel = bool(is_pixel)
        self.device = torch.device(device)

        # JVP settings
        self.ntk_samples = int(ntk_samples)
        self.jvp_chunk_size = max(1, int(jvp_chunk_size))

        # Checkpointing config
        self.use_checkpointing = bool(use_checkpointing)
        self.cp_critic = self.use_checkpointing if checkpoint_critic is None else bool(checkpoint_critic)
        self.cp_jacproj = self.use_checkpointing if checkpoint_jacproj is None else bool(checkpoint_jacproj)
        self.cp_latproj = self.use_checkpointing if checkpoint_latproj is None else bool(checkpoint_latproj)

        self.robust_marginals = bool(robust_marginals)
        self.eps = float(eps)

        # ---- One-time feature probe (no grads; not doing JVP here) ----
        self.decoder.eval()
        self.feature_extractor.eval()
        with torch.no_grad():
            dummy_z = torch.zeros(1, self.latent_dim, device=self.device)
            dummy_x = self.decoder(dummy_z)  # [1, D] or [1, C, H, W]
            if self.is_pixel:
                f = self.feature_extractor(dummy_x).reshape(1, -1)
            else:
                f = self.feature_extractor(dummy_x.reshape(1, -1)).reshape(1, -1)
            self._feat_dim = int(f.shape[-1])

        jacobian_dim = self.ntk_samples * self._feat_dim

        # Projectors + critic (cheap ReLU; no AMP)
        self.jacobian_projector = nn.Sequential(
            nn.Linear(jacobian_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, jac_dim),
        )
        self.latent_projector = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_proj_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(jac_dim + latent_proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Logging helpers
        self.register_buffer("running_mean", torch.tensor(0.0), persistent=True)
        self.register_buffer("ema_decay", torch.tensor(0.99), persistent=True)

        # Default back to training mode for user modules
        self.train()
        self.to(self.device)

    # ---------- utils ----------
    def to(self, device: Union[str, torch.device]):  # type: ignore[override]
        ret = super().to(device)
        self.device = torch.device(device)
        return ret

    @staticmethod
    def _maybe_checkpoint(mod: nn.Module, x: torch.Tensor, use_cp: bool) -> torch.Tensor:
        """
        Checkpoint only if requested AND at least one input requires grad.
        (PyTorch checkpoint needs a grad-requiring tensor input.)
        """
        if use_cp and isinstance(x, torch.Tensor) and x.requires_grad:
            return cp.checkpoint(mod, x, use_reentrant=False)
        return mod(x)

    def _encode_obs_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: [N, B, *] JVP outputs; returns [N, B, F]. No grads through encoder.
        """
        N, B = obs.shape[:2]
        if self.is_pixel:
            x = obs.reshape(N * B, *obs.shape[2:])
            with torch.inference_mode():
                feats = self.feature_extractor(x).reshape(N, B, -1)
        else:
            x = obs.reshape(N * B, -1)
            with torch.inference_mode():
                feats = self.feature_extractor(x).reshape(N, B, -1)
        return feats

    # ---------- JVP via torch.func.linearize (chunked tangents) ----------
    def _compute_chunked_jvp(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, Dz]  ->  returns [B, ntk_samples * F]
        Uses `linearize(self.decoder, z)` once, then applies jvp_fn to batched directions.
        """
        B = z.shape[0]
        N = self.ntk_samples
        F = self._feat_dim

        out = z.new_zeros(B, N * F)

        # Random unit directions (N, B, Dz)
        dirs = torch.randn(N, B, self.latent_dim, device=self.device)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

        # Make sure z is contiguous for the linearization
        z = z.contiguous()

        # Linearize decoder at z once; jvp_fn(v) returns J(z) @ v (same shape as decoder(z))
        # Do not wrap in no_grad; we want the primal trace for JVP.
        _, jvp_fn = linearize(self.decoder, z)

        # Keep feature_extractor stable (no BN/Dropout drift)
        was_train = self.feature_extractor.training
        self.feature_extractor.eval()

        start = 0
        while start < N:
            end = min(start + self.jvp_chunk_size, N)
            v_chunk = dirs[start:end].contiguous()  # [n, B, Dz]
            n = v_chunk.shape[0]

            # Apply JVP per direction (sequential to cap memory)
            jvp_list = []
            for i in range(n):
                v_in = v_chunk[i]  # [B, Dz]
                tangent = jvp_fn(v_in)  # [B, *obs]
                jvp_list.append(tangent)

            # Stack to [n, B, *obs] then encode to features
            Jv = torch.stack(jvp_list, dim=0)
            feats_nbf = self._encode_obs_features(Jv)                   # [n, B, F]
            block = feats_nbf.permute(1, 0, 2).reshape(B, n * F)        # [B, n*F]
            out[:, start * F : start * F + n * F] = block

            start = end
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if was_train:
            self.feature_extractor.train()
        del dirs
        return out  # [B, N*F]

    # ---------- Forward: MINE-DV lower bound ----------
    def forward(
        self,
        z_samples: torch.Tensor,   # [B, Dz]
        num_samples: int = 5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        I ≈ E_joint[T] - log E_marg[exp(T)]
        Returns per-batch scalar replicated to [B].
        """

        # Sample z ~ N(mean, diag(exp(logvar)))
        B, Dz = z_samples.shape
        if num_samples > 1:
            # Tile the latents S times along a new axis, then flatten
            z_samples = (
                z_samples.unsqueeze(1)               # [B, 1, Dz]
                        .expand(B, num_samples, Dz) # [B, S, Dz]  (no copy)
                        .reshape(num_samples * B, Dz)  # [S*B, Dz]
                        .contiguous()
    )
        
        z_samples = z_samples + torch.randn_like(z_samples)  # [S*B, Dz]

        # JVP features → projector (checkpoint if enabled & grad flows)
        jac_feats = self._compute_chunked_jvp(z_samples)                        # [S*B, N*F]
        jac_proj  = self._maybe_checkpoint(self.jacobian_projector,
                                           jac_feats, self.cp_jacproj)      # [S*B, jac_dim]

        # Latent projection (with grads)
        lat_proj  = self._maybe_checkpoint(self.latent_projector,
                                           z_samples, self.cp_latproj)          # [S*B, lat_dim]

        # Critic on joint pairs
        joint_in  = torch.cat([jac_proj, lat_proj], dim=-1)                  # [S*B, *]
        t_joint   = self._maybe_checkpoint(self.critic, joint_in, self.cp_critic).squeeze(-1)

        # Critic on marginal pairs (single perm or product of marginals)
        def _marg_block(jac, lat):
            # jac, lat: [S*B, D]
            if self.robust_marginals:
                jac_perm = torch.randperm(jac.shape[0], device=jac.device)
                lat_perm = torch.randperm(lat.shape[0], device=lat.device)
                m = torch.cat([jac[jac_perm], lat[lat_perm]], dim=-1)
            else:
                perm = torch.randperm(jac.shape[0], device=jac.device)
                m = torch.cat([jac, lat[perm]], dim=-1)
            return self.critic(m).squeeze(-1)

        if self.cp_critic:
            # use_reentrant=False preserves RNG state for recomputation
            t_marg = cp.checkpoint(_marg_block, jac_proj, lat_proj, use_reentrant=False)
        else:
            t_marg = _marg_block(jac_proj, lat_proj)

        # DV bound (stable)
        t_joint_mean      = t_joint.mean()
        t_marg_logmeanexp = torch.logsumexp(t_marg, dim=0) - math.log(t_marg.numel() + self.eps)
        mi_lower_bound    = t_joint_mean - t_marg_logmeanexp

        # EMA for logging
        self.running_mean = self.ema_decay * self.running_mean + (1 - self.ema_decay) * mi_lower_bound.detach()

        epistemic_value = mi_lower_bound.expand(B)
        del  z_samples, jac_feats, jac_proj, lat_proj, joint_in, t_joint, t_marg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        metrics = {
            "epistemic/mi_estimate": float(mi_lower_bound.detach().cpu()),
            "epistemic/joint_mean": float(t_joint_mean.detach().cpu()),
            "epistemic/marg_logmeanexp": float(t_marg_logmeanexp.detach().cpu()),
            "epistemic/running_mean": float(self.running_mean.detach().cpu()),
            "epistemic/ntk_samples": self.ntk_samples,
            "epistemic/jvp_chunk_size": self.jvp_chunk_size,
            "epistemic/feat_dim": self._feat_dim,
            "epistemic/checkpoint": int(self.use_checkpointing),
            "epistemic/robust_marginals": int(self.robust_marginals),
        }
        return torch.clamp(epistemic_value, min=0.0), metrics
