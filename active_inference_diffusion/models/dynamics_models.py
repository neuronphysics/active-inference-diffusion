"""
Dynamics model implementations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch.nn.attention import sdpa_kernel
class LatentDynamicsModel(nn.Module):
    """
    Latent dynamics model f([state_t, action_t]) -> state_{t+1}
    LSTM variant with optional per-item reset via `done_mask` (like the Transformer).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        residual: bool = True,
        lstm_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.residual = residual
        self.lstm_hidden_dim = lstm_hidden_dim

        # Belief LSTM over [s_t, a_t]; we do single-step updates (seq_len=1) with batch_first=True
        self.belief_lstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # (Optional) buffers if you later want learned h0/c0; currently unused (zeros)
        self.register_buffer("lstm_h0", None)
        self.register_buffer("lstm_c0", None)

        # MLP that reads [s_t, a_t, h_t] and outputs (mean, logvar) for s_{t+1}
        input_dim = state_dim + action_dim + lstm_hidden_dim
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 2 * state_dim))
        self.network = nn.Sequential(*layers)

        # Output init: tiny if residual, otherwise Xavier
        if residual:
            nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
            nn.init.zeros_(self.network[-1].bias)
        else:
            nn.init.xavier_uniform_(self.network[-1].weight, gain=1.0)
            nn.init.zeros_(self.network[-1].bias)

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM weights for stable training."""
        for name, param in self.belief_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Forget gate bias = 1 (gates order: i, f, g, o)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh hidden for new sequences. Shape: (num_layers, B, lstm_hidden_dim)."""
        num_layers = self.belief_lstm.num_layers
        h0 = torch.zeros(num_layers, batch_size, self.lstm_hidden_dim, device=device)
        c0 = torch.zeros(num_layers, batch_size, self.lstm_hidden_dim, device=device)
        return (h0, c0)

    def _maybe_reinit_hidden(
        self,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_state is None:
            return self.init_hidden(batch_size, device)
        h, c = hidden_state
        # If batch size changed (e.g., at episode start), reinit
        if h.size(1) != batch_size or c.size(1) != batch_size:
            return self.init_hidden(batch_size, device)
        return h, c

    def forward(
        self,
        state: torch.Tensor,                              # (B, state_dim)
        action: torch.Tensor,                             # (B, action_dim)
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        done_mask: Optional[torch.Tensor] = None          # (B,) bool or {0,1}; True resets BEFORE current step
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns:
            next_state_mean: (B, state_dim)
            next_state_logvar: (B, state_dim)
            new_hidden_state: (h, c) where each is (num_layers, B, lstm_hidden_dim)
        """
        B = state.size(0)
        device = state.device

        # Prepare input and hidden
        inputs = torch.cat([state, action], dim=-1)              # (B, S+A)
        if hidden_state is None:
            hidden_state = self.init_hidden(B, device)

        if done_mask is not None:
            dm = done_mask.float().view(1, B, 1)  # [1, B, 1] for broadcasting
            h, c = hidden_state
            # h,c: [num_layers, B, H] - broadcast correctly
            h = h * (1.0 - dm)  # No transpose needed
            c = c * (1.0 - dm)
            hidden_state = (h, c)

        lstm_out, new_hidden = self.belief_lstm(inputs.unsqueeze(1), hidden_state)

        combined_feature = torch.cat([inputs, new_hidden[0][-1]], dim=-1)
        output = self.network(combined_feature)
        mean_state, log_var_state = torch.chunk(output, 2, dim=-1)

        next_state_mean = state + mean_state if self.residual else mean_state
        next_state_logvar = torch.clamp(log_var_state, min=-10, max=2)
        return next_state_mean, next_state_logvar, new_hidden

class TransformerDynamicsModel(nn.Module):
    """
    Causal Transformer dynamics: f([state, action]_t, context) -> state_{t+1}
    
      - init_hidden(batch, device) -> hidden
      - forward(state, action, hidden, done_mask=None) -> (mean, logvar, hidden)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,     # d_model
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        context_len: int = 16,
        residual: bool = True,
        use_checkpointing: bool = False,
        attn_impl: str = "auto",   # {"auto","flash","mem","math"} for SDPA backends
        clear_on_reset: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = hidden_dim
        self.context_len = context_len
        self.residual = residual
        self.use_checkpointing = use_checkpointing
        self.attn_impl = attn_impl
        self.clear_on_reset = clear_on_reset
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Token for [state, action]
        self.token_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learned absolute positions [T, d_model]
        self.pos_embed = nn.Parameter(torch.zeros(context_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Pre-norm Transformer encoder (batch_first=True)
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

        # Head: (mean, logvar) of next state
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * state_dim),
        )
        nn.init.uniform_(self.head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head[-1].bias)


    def init_hidden(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        # Fixed-capacity ring buffer to avoid reallocation each step
        tokens = torch.empty(batch_size, self.context_len, self.d_model, device=device)
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)  # valid length in [0..context_len]
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
        # (T,T) with -inf above diagonal
        return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)

    @staticmethod
    def _key_padding_mask(lengths: torch.Tensor, T: int):
        """
        lengths: (B,) valid lengths, 0..T
        return: (B, T) True where PAD, False where valid
        """
        B = lengths.size(0)
        ar = torch.arange(T, device=lengths.device).unsqueeze(0).expand(B, T)  # (B,T)
        return ar >= lengths.clamp_max(T).unsqueeze(1)

    def _sdpa_backend_ctx(self):
        """
        Select SDPA backend if on CUDA + PyTorch>=2.
        - flash: fastest (constraints on head dim etc.)
        - mem: memory-efficient
        - math: matmul
        - auto: let PyTorch decide
        """
        try:
            
            if self.attn_impl == "flash":
                return sdpa_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
            elif self.attn_impl == "mem":
                return sdpa_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
            elif self.attn_impl == "math":
                return sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            else:  # "auto"
                class _NullCtx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                return _NullCtx()
        except Exception:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            return _NullCtx()

    def get_memory_usage(self, batch_size: int) -> Dict[str, float]:
        """Rough memory usage report (MB) for the token buffer + parameters (fp32)."""
        token_mb = batch_size * self.context_len * self.d_model * 4 / 1024**2
        param_mb = sum(p.numel() * 4 for p in self.parameters()) / 1024**2
        return {"token_buffer_mb": token_mb, "parameters_mb": param_mb, "total_mb": token_mb + param_mb}


    def forward(
        self,
        state: torch.Tensor,                     # (B, S)
        action: torch.Tensor,                    # (B, A)
        hidden_state: Optional[Dict[str, torch.Tensor]] = None,
        done_mask: Optional[torch.Tensor] = None # (B,) bool or 0/1; True resets sequence BEFORE appending current token
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B = state.size(0)
        device = state.device

        hidden_state = self._maybe_reset_hidden(hidden_state, B, device)
        tokens = hidden_state["tokens"]        # (B, Tcap, d)
        lengths = hidden_state["lengths"]      # (B,)

        if done_mask is not None and bool(done_mask.any().item()):

            idx = done_mask.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                tokens[idx].zero_()
                if lengths is not None:
                    lengths[idx] = 0

        # Project current [s_t, a_t] into token
        token = self.token_proj(torch.cat([state, action], dim=-1))  # (B, d_model)

        # Append to ring buffer (shift-left where full)
        full_mask = lengths >= self.context_len
        if full_mask.any():
            rows = full_mask.nonzero(as_tuple=False).squeeze(-1)
            tokens[rows, :-1, :] = tokens[rows, 1:, :]
            tokens[rows, -1, :] = token[rows]
        if (~full_mask).any():
            rows = (~full_mask).nonzero(as_tuple=False).squeeze(-1)
            pos = lengths[rows]
            tokens[rows, pos, :] = token[rows]
            lengths[rows] = pos + 1

        # Effective sequence length
        T = int(lengths.max().item())
        T = max(1, min(T, self.context_len))

        # Slice valid window and build masks
        x = tokens[:, :T, :]                          # (B, T, d)
        x = x + self.pos_embed[:T, :].unsqueeze(0)
        src_kpm = self._key_padding_mask(lengths, T)  # (B, T) True=pad
        causal = self._causal_mask(T, device)         # (T, T)

        # Encoder
        with self._sdpa_backend_ctx():
            if self.use_checkpointing:
                amp_enabled = torch.is_autocast_enabled()
                amp_dtype = None
                try:
                    amp_dtype = torch.get_autocast_gpu_dtype()
                except Exception:
                    pass
              
                def run_layer(y, layer, cm, kpm):
                    # Match autocast state exactly during recompute
                    device_type = "cuda" if y.is_cuda else "cpu"
                    with torch.amp.autocast(device_type=device_type,
                                            enabled=amp_enabled,
                                            dtype=(amp_dtype if device_type == "cuda" else None)):
                         # Boolean src_mask; do NOT pass is_causal to avoid path switches
                        return layer(y, src_mask=cm, src_key_padding_mask=kpm)

                
                for layer in self.encoder.layers:
                    x = torch.utils.checkpoint.checkpoint(
                        run_layer, x, layer, causal, src_kpm, use_reentrant=False, preserve_rng_state=True
                    )

                if self.encoder.norm is not None:
                    x = self.encoder.norm(x)
            else:
                # Also use KWARGS here for stability across versions
                x = self.encoder(x, mask=causal, src_key_padding_mask=src_kpm)
        # Last VALID token per item
        idx = (lengths - 1).clamp(min=0, max=T - 1)          # (B,)
        h = x[torch.arange(B, device=device), idx, :]        # (B, d)

        # Predict next state distribution
        out = self.head(h)                                   # (B, 2*S)
        mean_state, log_var_state = torch.chunk(out, 2, dim=-1)

        next_state_mean = state + mean_state if self.residual else mean_state
        next_state_logvar = torch.clamp(log_var_state, min=self.logvar_min, max=self.logvar_max)

        new_hidden = {"tokens": tokens, "lengths": lengths}
        return next_state_mean, next_state_logvar, new_hidden
