#
from typing import Optional, Tuple, List, Dict
from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor

EPS_TINY = 1e-6

class BaseDivergence(nn.Module):
    """Base class for all divergence measures"""

    def __init__(self, nclass: int, param: Optional[List[float]] = None,
                 softmax_logits: bool = True, softmax_gt: bool = True):
        """
        Initialize base divergence measure.

        Args:
            nclass: Number of classes
            param: Optional parameters for specific divergence measures
            softmax_logits: Whether to apply softmax to model predictions
            softmax_gt: Whether to apply softmax to ground truth distributions
        """
        super(BaseDivergence, self).__init__()
        self.nclass = nclass
        self.param = [] if param is None else param
        self.softmax_logits = softmax_logits
        self.softmax_gt = softmax_gt
        assert nclass >= 2, "Number of classes must be at least 2"

    def prepare_inputs(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process inputs based on their shapes.

        Args:
            logits: Model predictions
            targets: Ground truth (class indices or distributions)

        Returns:
            Processed logits and targets
        """
        if len(targets.shape) == len(logits.shape) - 1:
            # If targets are class indices, convert to one-hot
            targets = F.one_hot(targets, self.nclass).to(logits.dtype).to(logits.device)

            # If targets are already distributions
        if self.softmax_logits:
            logits = F.softmax(logits, dim=1)

        # Only apply softmax to targets if they're not already one-hot encoded and softmax_gt is True
        if self.softmax_gt and not torch.allclose(targets.sum(dim=1),
                                                  torch.ones(targets.size(0), device=targets.device)):
            targets = F.softmax(targets, dim=1)

        return logits, targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """To be implemented by child classes"""
        raise NotImplementedError

class fDivergence(BaseDivergence):
    """
    General f-Divergence with customizable f function

    The f-divergence is defined as:
    D_f(P||Q) = ∑ q(x) * f(p(x)/q(x))

    where f is a convex function satisfying f(1) = 0
    """

    def __init__(self, 
                 nclass: int, 
                 param: Optional[List[float]] = None,
                 softmax_logits: bool = True, 
                 softmax_gt: bool = True,
                 f_func: Optional[callable] = lambda x: -torch.log(x) #Reverse KL by default
                 ):
        super(fDivergence, self).__init__(nclass, param, softmax_logits, softmax_gt)
        # Default to GAN if no function provided
        self.f_func = f_func if f_func is not None else lambda x: x * torch.log(x) - (x + 1)*torch.log(x + 1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self.prepare_inputs(logits, targets)
        # Compute ratio with numerical stability
        safe_logits = torch.clamp(logits, min=EPS_TINY)
        ratio = torch.clamp(targets / safe_logits, min=EPS_TINY)
        # Compute f-divergence: D_f(P||Q) = ∑ q(x) * f(p(x)/q(x))
        f_values = self.f_func(ratio)
        return (logits * f_values).sum(dim=1).mean()

class DVKL(nn.Module):
    def __init__(self, 
                 T_network: nn.Module, 
                 ema_momentum: float = 0.99,
                 temperature: float = 1.0, 
                 temperature_anneal: float = 0.999):
        super().__init__()
        self.T = T_network
        self.ema_momentum = ema_momentum
        self.register_buffer('ema_Eexp', torch.tensor(0.0), persistent=True)
        self.register_buffer('ema_initialized', torch.tensor(False), persistent=True)
        self.register_buffer('temperature', torch.tensor(temperature), persistent=True)
        self.temperature_anneal = temperature_anneal

    @torch.no_grad()
    def anneal_temperature(self):
        self.temperature.mul_(self.temperature_anneal)
        self.temperature.clamp_(min=0.1)

    def _dv_second(self, T_p: torch.Tensor) -> torch.Tensor:
        # log E_p[e^{T}] via log-mean-exp; keep EMA if you were using it
        B = T_p.shape[0]
        return T_p.logsumexp(dim=0) - torch.log(torch.tensor(B, device=T_p.device))

    def forward(self, z_q: torch.Tensor, prior) -> torch.Tensor:
        # First term: posterior
        T_q = self.T(z_q).squeeze(-1)                       # (B,)

        # Second term: prior with pathwise gradients into ψ
        z_p, _ = prior.sample(B=z_q.size(0),
                              temperature=float(self.temperature),
                              hard=False,
                              device=z_q.device)
        T_p = self.T(z_p).squeeze(-1)                       # (B,)
        # DV bound: KL >= E_q[T] - log E_p[exp(T)]
        return T_q.mean() - self._dv_second(T_p)
    
