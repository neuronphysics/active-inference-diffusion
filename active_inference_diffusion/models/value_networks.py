"""
Value network implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from active_inference_diffusion.utils.util import (
    make_symlog_bins, twohot_encode, symlog, symexp, categorical_ce_with_soft_targets
)

class ValueNetwork(nn.Module):
    """
    Dreamer-style critic: categorical distribution over symlog-spaced bins.
    No time conditioning; predicts V(R | s) as logits over K bins.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3, num_bins: int = 255):
        super().__init__()
        self.num_bins = int(num_bins)

        layers = []
        input_dim = state_dim
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)

        self.head = nn.Linear(hidden_dim, self.num_bins)
        # Dreamer recommends zero-init output head to avoid large early predictions.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Precompute bins in symlog & real space for fast converts
        bins_symlog, bins_real = make_symlog_bins(self.num_bins)  # util.py
        self.register_buffer("bins_symlog", bins_symlog)  # [K]
        self.register_buffer("bins_real", bins_real)      # [K]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.backbone(state)
        return self.head(h)  # [B, K] logits

    @torch.no_grad()
    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        # Convert categorical to expectation in REAL space for bootstrapping/actor
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bins_real).sum(dim=-1)  # [B]

    def loss_from_returns(self, logits: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        returns: [B] real-valued λ-returns
        """
        targets_symlog = symlog(returns)                       # util.py
        soft_targets = twohot_encode(targets_symlog, self.bins_symlog)  # util.py
        return categorical_ce_with_soft_targets(logits, soft_targets)   # util.py
