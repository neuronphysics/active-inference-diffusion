"""
Value network implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from active_inference_diffusion.utils.util import DiscDist

class ValueNetwork(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 hidden_dim: int = 256, 
                 num_layers: int = 3, 
                 num_bins: int = 255):
        super().__init__()
        self.num_bins = int(num_bins)
        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.ReLU()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.num_bins)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        # store range once
        self.low, self.high = -10.0, 10.0

    def forward(self, state):
        return self.head(self.backbone(state))  # [B, K]

    @torch.no_grad()
    def expected_value(self, logits):
        dist = DiscDist(logits, low=self.low, high=self.high, device=logits.device)
        return dist.mean().squeeze(-1)  # <- squeeze to [B]

    def loss_from_returns(self, logits, returns):
        dist = DiscDist(logits, low=self.low, high=self.high, device=logits.device)
        return (-dist.log_prob(returns)).mean()   # returns can be [B]

