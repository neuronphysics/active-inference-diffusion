"""
Free Energy computation module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class FreeEnergyComputation(nn.Module):
    """
    F = D_KL[q(z)||p(z)] - E_q[log p(o|z)]
      = Complexity - Accuracy
    """
    def __init__(self,
                 precision_init: float = 1.0,
                 observation_decoder: Optional[nn.Module] = None,
                 is_pixel_observation: bool = False):
        super().__init__()
        self.log_precision = nn.Parameter(torch.log(torch.tensor(precision_init)))
        self.observation_decoder = observation_decoder
        self.is_pixel_observation = is_pixel_observation

    @property
    def precision(self) -> torch.Tensor:
        return torch.exp(self.log_precision)

    def set_decoder(self, decoder: nn.Module):
        self.observation_decoder = decoder

    def compute_loss(
        self,
        states: torch.Tensor,              # z
        observations: torch.Tensor,        # o (features) OR raw pixels if state-based task
        score_network: nn.Module,
        current_time: float = 0.0,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: float = 1.0,
        raw_observations: Optional[torch.Tensor] = None,  # required if is_pixel_observation & pixel loss
        frame_idx: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if prior_mean is None:
            prior_mean = torch.zeros_like(states)

        B = states.shape[0]
        device = states.device

        # Complexity term (unit-variance prior for simplicity)
        complexity = 0.5 * ((states - prior_mean) ** 2 / (prior_std ** 2)).sum(dim=-1).mean()

        # ---- Accuracy term: decode z -> observation space, then compare to targets ----
        if self.observation_decoder is None:
            raise ValueError("FreeEnergyComputation: observation_decoder is not set.")

        decoded = self.observation_decoder(states)

        if self.is_pixel_observation and raw_observations is not None:
            # Compare in pixel space (B,C,H,W)
            target = raw_observations
            if decoded.shape != target.shape:
                # fallback: flatten if shapes differ
                decoded = decoded.flatten(1)
                target  = target.flatten(1)
            obs_err = F.mse_loss(decoded, target, reduction='none')
            obs_err = obs_err.view(B, -1).mean(dim=1)        # per-sample MSE
        else:
            # Compare to feature/state observations (B, D)
            if decoded.dim() > 2:
                decoded = decoded.flatten(1)
            if observations.dim() > 2:
                observations = observations.flatten(1)
            obs_err = F.mse_loss(decoded, observations, reduction='none').mean(dim=1)

        accuracy = -0.5 * self.precision * obs_err.mean()

        # Score regularizer (keep as in your pipeline)
        t = torch.full((B,), current_time, device=device)
        score = score_network(states, t, observations, frame_time=frame_idx, action=actions)
        score_reg = 0.01 * (score ** 2).sum(dim=-1).mean()

        free_energy = complexity - accuracy + score_reg

        info = {
            "complexity": complexity.detach(),
            "accuracy": (-accuracy).detach(),            # positive for logging
            "observation_error": obs_err.mean().detach(),
            "score_regularization": score_reg.detach(),
            "precision": self.precision.detach(),
            "reconstruction_mse": obs_err.mean().detach(),
        }
        return free_energy, info

    def update_precision(self, complexity: torch.Tensor, accuracy: torch.Tensor):
        # Decrease precision when complexity > accuracy (sign fix)
        with torch.no_grad():
            precision_error = complexity - accuracy
            self.log_precision.data -= 0.01 * precision_error.clamp(-1, 1)
            self.log_precision.data.clamp_(-3, 3)
