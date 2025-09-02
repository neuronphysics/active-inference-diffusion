"""
Dynamics model implementations
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

# ---------- Conditional Grassmann-Mixture prior (batch-wise) ----------

class ConditionalGrassmannMixture:
    """
    Batch-conditional mixture on a Grassmann manifold (per-sample parameters).

    For each item b in the batch, we have a K-component mixture with:
      - logits[b, K]
      - U_raw[b, K, D, r]   (orthonormalized via QR -> Stiefel bases)
      - m[b, K, r]          (means in subspace coordinates)
      - log_sigma_par[b, K], log_sigma_perp[b, K]

    API mirrors your existing prior:
      - sample(B, temperature, hard, device) -> (z: [B, D], y: [B, K])
      - log_prob(z) (optional, for debugging)
    """

    def __init__(
        self,
        logits: torch.Tensor,           # (B, K)
        U_raw: torch.Tensor,            # (B, K, D, r)
        m: torch.Tensor,                # (B, K, r)
        log_sigma_par: torch.Tensor,    # (B, K)
        log_sigma_perp: torch.Tensor,   # (B, K)
    ):
        assert logits.dim() == 2
        assert U_raw.dim() == 4
        B, K = logits.shape
        assert U_raw.shape[:2] == (B, K)
        self.logits = logits
        self.U_raw = U_raw
        self.m = m
        self.log_sigma_par = log_sigma_par
        self.log_sigma_perp = log_sigma_perp
        self.B = B
        self.K = K
        self.D = U_raw.shape[2]
        self.r = U_raw.shape[3]
        self.device = logits.device
        self.dtype = logits.dtype

    def _orthonormalize(self) -> torch.Tensor:
        """
        QR per (B,K) -> U with orthonormal columns.
        torch.linalg.qr supports batching over leading dims.
        Returns: U of shape (B, K, D, r)
        """
        # Flatten (B,K) for a single batched QR, then unflatten.
        BK, D, r = self.B * self.K, self.D, self.r
        U_in = self.U_raw.reshape(BK, D, r)
        Q, R = torch.linalg.qr(U_in)  # (BK, D, r)
        # Fix sign ambiguity for stability (match your prior).
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        s = torch.sign(diag)
        s = torch.where(s == 0, torch.ones_like(s), s)
        Q = Q * s.unsqueeze(-2)
        return Q.reshape(self.B, self.K, D, r)

    @torch.no_grad()
    def _project_perp(self, U: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Project x onto the orthogonal complement of span(U).
        U: (B,K,D,r), x: (B,K,D) -> x_perp: (B,K,D)
        """
        # e_par = U^T x  -> (B,K,r)
        e_par = torch.einsum('bkdr,bkd->bkr', U, x)
        # P e = U e_par   -> (B,K,D)
        Pe = torch.einsum('bkdr,bkr->bkd', U, e_par)
        return x - Pe

    def sample(
        self,
        B: Optional[int] = None,
        temperature: float = 0.7,
        hard: bool = False,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable mixture draw z ~ sum_k y_k N_k(·) with:
          1) y ~ Concrete(pi, tau)
          2) For each k, draw z_k = U_k (m_k + σ∥ ε∥) + σ⊥ (I - U_kU_k^T) ε⊥
          3) z = Σ_k y_k z_k
        Returns:
          z: (B, D), y: (B, K)
        Matches your global prior's API so DV-KL can call it directly.  :contentReference[oaicite:3]{index=3}
        """
        if B is not None:
            # Sanity: we return exactly one sample per conditional prior in the batch.
            if B != self.B:
                raise ValueError(f"Conditional prior is batch-sized ({self.B}); got B={B}.")
        B = self.B
        device = device or self.device

        K, D, r = self.K, self.D, self.r
        U = self._orthonormalize()  # (B,K,D,r)

        # 1) Gumbel-Softmax over mixture logits (pathwise)
        dist = RelaxedOneHotCategorical(
            temperature=torch.as_tensor(temperature, device=device, dtype=self.dtype),
            logits=self.logits,
        )
        y = dist.rsample()  # (B,K)
        if hard:
            idx = y.argmax(dim=-1)
            y_h = F.one_hot(idx, num_classes=K).to(y.dtype)
            y = (y_h - y).detach() + y  # straight-through

        # 2) Per-component samples
        eps_par = torch.randn(B, K, r, device=device, dtype=self.dtype)           # ε∥
        eps_perp = torch.randn(B, K, D, device=device, dtype=self.dtype)          # ε⊥

        sig_par = torch.exp(self.log_sigma_par)[:, :, None]    # (B,K,1)
        sig_perp = torch.exp(self.log_sigma_perp)[:, :, None]  # (B,K,1)

        z_par = self.m + sig_par * eps_par                     # (B,K,r)
        z_par_amb = torch.einsum('bkdr,bkr->bkd', U, z_par)    # U z∥  -> (B,K,D)

        e_perp = self._project_perp(U, eps_perp)               # (I - UU^T) ε⊥
        z_perp = sig_perp * e_perp                              # (B,K,D)

        z_k = z_par_amb + z_perp                                # (B,K,D)

        # 3) Blend by y
        z = (y.unsqueeze(-1) * z_k).sum(dim=1)                  # (B,D)
        return z, y

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Optional: log p(z) = logsumexp_k [ log pi_k + log N_k(z) ].
        Useful for monitoring but not required by DV-KL.  :contentReference[oaicite:4]{index=4}
        """
        B, D = z.shape
        assert B == self.B and D == self.D
        U = self._orthonormalize()                              # (B,K,D,r)

        # Projections
        z_par = torch.einsum('bd,bkdr->bkr', z, U)              # (B,K,r)
        z2 = (z * z).sum(dim=-1, keepdim=True)                  # (B,1)
        zpar2 = (z_par * z_par).sum(dim=-1)                     # (B,K)
        zperp2 = (z2 - zpar2).clamp_min(0.)                     # (B,K)

        logsp = self.log_sigma_par                              # (B,K)
        logso = self.log_sigma_perp                             # (B,K)
        invp = torch.exp(-2.0 * logsp)
        invo = torch.exp(-2.0 * logso)

        spar = ((z_par - self.m) ** 2).sum(dim=-1)              # (B,K)

        rdim = U.shape[-1]
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
        log_norm = -0.5 * (rdim * (log_2pi + 2 * logsp) + (D - rdim) * (log_2pi + 2 * logso))  # (B,K)
        logNk = log_norm - 0.5 * (spar * invp + zperp2 * invo)                                   # (B,K)
        log_pi = F.log_softmax(self.logits, dim=-1)                                               # (B,K)
        return torch.logsumexp(log_pi + logNk, dim=-1)                                            # (B,)

# ---------- Shared small MLP block ----------

def mlp(in_dim, hidden, out_dim, num_layers=2, act=nn.ELU, dropout=0.0):
    layers = []
    for i in range(num_layers):
        layers += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.LayerNorm(hidden), act(), nn.Dropout(dropout)]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)

# ---------- LSTM Grassmann dynamics ----------

class LatentDynamicsModel(nn.Module):
    """
    f([z_t, a_t], h_t) -> ConditionalGrassmannMixture over z_{t+1}

    Returns:
      prior: ConditionalGrassmannMixture
      new_hidden: (h, c)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        K: int,
        r: int,
        lstm_hidden: int = 128,
        trunk_hidden: int = 256,
        trunk_layers: int = 2,
        dropout: float = 0.1,
        log_sigma_bounds: Tuple[float, float] = (-3.0, 1.5),  # ~[0.05, 4.5]
    ):
        super().__init__()
        self.D = state_dim
        self.A = action_dim
        self.K = K
        self.r = r
        self.logsig_min, self.logsig_max = log_sigma_bounds

        self.belief_lstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        feat_dim = state_dim + action_dim + lstm_hidden
        self.trunk = mlp(feat_dim, trunk_hidden, trunk_hidden, num_layers=trunk_layers, dropout=dropout)

        # Separate heads for mixture parameters
        self.head_logits = nn.Linear(trunk_hidden, K)
        self.head_U = nn.Linear(trunk_hidden, K * state_dim * r)
        self.head_m = nn.Linear(trunk_hidden, K * r)
        self.head_logs = nn.Linear(trunk_hidden, 2 * K)

        # Small init to keep things stable
        for head in [self.head_logits, self.head_U, self.head_m, self.head_logs]:
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            nn.init.zeros_(head.bias)

        # LSTM init
        for name, p in self.belief_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias

    def init_hidden(self, B: int, device: torch.device):
        h0 = torch.zeros(2, B, self.belief_lstm.hidden_size, device=device)
        c0 = torch.zeros(2, B, self.belief_lstm.hidden_size, device=device)
        return (h0, c0)

    def forward(
        self,
        state: torch.Tensor,                 # (B, D)
        action: torch.Tensor,                # (B, A)
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done_mask: Optional[torch.Tensor] = None,   # (B,), True resets BEFORE current step
    ) -> Tuple[ConditionalGrassmannMixture, Tuple[torch.Tensor, torch.Tensor]]:
        B = state.size(0)
        device = state.device

        x = torch.cat([state, action], dim=-1)           # (B, D+A)
        if hidden is None:
            hidden = self.init_hidden(B, device)

        if done_mask is not None and done_mask.any():
            dm = done_mask.float().view(1, B, 1)
            h, c = hidden
            h = h * (1.0 - dm)
            c = c * (1.0 - dm)
            hidden = (h, c)

        _, new_hidden = self.belief_lstm(x.unsqueeze(1), hidden)   # single-step
        h_last = new_hidden[0][-1]                                 # (B, lstm_hidden)

        feat = self.trunk(torch.cat([x, h_last], dim=-1))          # (B, H)

        logits = self.head_logits(feat)                            # (B, K)
        U_raw = self.head_U(feat).view(B, self.K, self.D, self.r)  # (B, K, D, r)
        m = self.head_m(feat).view(B, self.K, self.r)              # (B, K, r)
        logs = self.head_logs(feat).view(B, self.K, 2)             # (B, K, 2)
        log_sigma_par = logs[..., 0].clamp_(self.logsig_min, self.logsig_max)   # (B, K)
        log_sigma_perp = logs[..., 1].clamp_(self.logsig_min, self.logsig_max)  # (B, K)

        prior = ConditionalGrassmannMixture(logits, U_raw, m, log_sigma_par, log_sigma_perp)
        return prior, new_hidden

# ---------- Transformer Grassmann dynamics ----------

class TransformerDynamicsModel(nn.Module):
    """
    Causal Transformer dynamics -> ConditionalGrassmannMixture over z_{t+1}.

    Keeps a context of the last `context_len` tokens of [z_t, a_t].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        K: int,
        r: int,
        hidden_dim: int = 256,     # d_model
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        context_len: int = 16,
        attn_impl: str = "auto",
        log_sigma_bounds: Tuple[float, float] = (-3.0, 1.5),
        clear_on_reset: bool = True,
    ):
        super().__init__()
        self.D = state_dim
        self.A = action_dim
        self.K = K
        self.r = r
        self.d_model = hidden_dim
        self.context_len = context_len
        self.attn_impl = attn_impl
        self.clear_on_reset = clear_on_reset
        self.logsig_min, self.logsig_max = log_sigma_bounds

        self.token_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(context_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Trunk + heads for mixture parameters
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_logits = nn.Linear(hidden_dim, K)
        self.head_U = nn.Linear(hidden_dim, K * state_dim * r)
        self.head_m = nn.Linear(hidden_dim, K * r)
        self.head_logs = nn.Linear(hidden_dim, 2 * K)

        for head in [self.head_logits, self.head_U, self.head_m, self.head_logs]:
            nn.init.uniform_(head.weight, -1e-3, 1e-3)
            nn.init.zeros_(head.bias)

    def init_hidden(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        tokens = torch.empty(B, self.context_len, self.d_model, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        tokens.zero_();  # start clean
        return {"tokens": tokens, "lengths": lengths}

    def _maybe_reset_hidden(self, hidden: Optional[Dict[str, torch.Tensor]], B: int, device):
        if (
            hidden is None
            or ("tokens" not in hidden)
            or hidden["tokens"].size(0) != B
            or hidden["tokens"].size(1) != self.context_len
            or hidden["tokens"].size(2) != self.d_model
        ):
            return self.init_hidden(B, device)
        if "lengths" not in hidden or hidden["lengths"].size(0) != B:
            hidden["lengths"] = torch.zeros(B, dtype=torch.long, device=device)
        return hidden

    @staticmethod
    def _causal_mask(T: int, device: torch.device):
        return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)

    @staticmethod
    def _key_padding_mask(lengths: torch.Tensor, T: int):
        B = lengths.size(0)
        ar = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)
        return ar >= lengths.clamp_max(T).unsqueeze(1)

    def forward(
        self,
        state: torch.Tensor,                     # (B, D)
        action: torch.Tensor,                    # (B, A)
        hidden: Optional[Dict[str, torch.Tensor]] = None,
        done_mask: Optional[torch.Tensor] = None # (B,)
    ) -> Tuple[ConditionalGrassmannMixture, Dict[str, torch.Tensor]]:
        B, device = state.size(0), state.device
        hidden = self._maybe_reset_hidden(hidden, B, device)
        tokens = hidden["tokens"]
        lengths = hidden["lengths"]

        if done_mask is not None and self.clear_on_reset and done_mask.any():
            idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                tokens[idx].zero_()
                lengths[idx] = 0

        tok = self.token_proj(torch.cat([state, action], dim=-1))  # (B, d_model)

        # Append with ring-buffer behaviour
        full = lengths >= self.context_len
        if full.any():
            rows = full.nonzero(as_tuple=False).squeeze(-1)
            tokens[rows, :-1, :] = tokens[rows, 1:, :]
            tokens[rows, -1, :] = tok[rows]
        if (~full).any():
            rows = (~full).nonzero(as_tuple=False).squeeze(-1)
            pos = lengths[rows]
            tokens[rows, pos, :] = tok[rows]
            lengths[rows] = pos + 1

        T = int(lengths.max().item())
        T = max(1, min(T, self.context_len))
        x = tokens[:, :T, :] + self.pos_embed[:T, :].unsqueeze(0)
        src_kpm = self._key_padding_mask(lengths, T)
        causal = self._causal_mask(T, device)

        # Encoder
        x = self.encoder(x, mask=causal, src_key_padding_mask=src_kpm)
        idx = (lengths - 1).clamp(min=0, max=T - 1)
        h = x[torch.arange(B, device=device), idx, :]           # (B, d_model)

        feat = self.trunk(h)

        logits = self.head_logits(feat)                          # (B, K)
        U_raw = self.head_U(feat).view(B, self.K, self.D, self.r)
        m = self.head_m(feat).view(B, self.K, self.r)
        logs = self.head_logs(feat).view(B, self.K, 2)
        log_sigma_par = logs[..., 0].clamp_(self.logsig_min, self.logsig_max)
        log_sigma_perp = logs[..., 1].clamp_(self.logsig_min, self.logsig_max)

        prior = ConditionalGrassmannMixture(logits, U_raw, m, log_sigma_par, log_sigma_perp)
        new_hidden = {"tokens": tokens, "lengths": lengths}
        return prior, new_hidden
