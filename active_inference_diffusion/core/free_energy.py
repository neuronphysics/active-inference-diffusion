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

    def compute_accuracy(
        self,
        states: torch.Tensor,              # z
        observations: torch.Tensor,        # o (features) OR raw pixels if state-based task
        raw_observations: Optional[torch.Tensor] = None,  # required if is_pixel_observation & pixel loss

    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        B = states.shape[0]
        device = states.device


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

        accuracy = 0.5 * self.precision * obs_err.mean()


        info = {
            "accuracy": (accuracy).detach(),            # positive for logging
            "observation_error": obs_err.mean().detach(),
            "precision": self.precision.detach(),
            "reconstruction_mse": obs_err.mean().detach(),
        }
        return accuracy, info

    def update_precision(self, complexity: torch.Tensor, accuracy: torch.Tensor):
        # Decrease precision when complexity > accuracy (sign fix)
        with torch.no_grad():
            precision_error = complexity - accuracy
            self.log_precision.data -= 0.01 * precision_error.clamp(-1, 1)
            self.log_precision.data.clamp_(-3, 3)
