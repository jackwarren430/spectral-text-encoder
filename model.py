import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(L: int, d_model: int, device, dtype):
    """Standard sin/cos positional encoding. Returns (1, L, d_model)."""
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even for sinusoidal PE, got {d_model}")
    position = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.empty(L, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def f_init_bias(L: int, d_sine: int, spread: float, device, dtype, real_lengths=None):
    """Per-(slot, channel) pre-sigmoid bias on the f channel.

    Without padding (real_lengths is None): one linspace over L·d_sine slots,
    same for every row. Slot 0 = -spread, slot L·d_sine-1 = +spread.

    With padding (real_lengths: (B,) long): per-row linspace over
    real_lengths[b]·d_sine slots so the bias for a sentence depends only on its
    own length, not on the batch's L_max. This is what makes a sentence's
    waveform invariant to which other (longer) sentences sit in the batch.
    Bias values for slots beyond real_lengths[b] are unused (A is zeroed by the
    pad mask before the synthesis sum)."""
    if real_lengths is None:
        return torch.linspace(-spread, spread, L * d_sine, device=device, dtype=dtype).view(
            1, L, d_sine
        )
    B = real_lengths.size(0)
    positions = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)
    channels = torch.arange(d_sine, device=device, dtype=dtype).view(1, 1, d_sine)
    slot_idx = positions * d_sine + channels  # (1, L, d_sine)
    denom = (real_lengths.to(dtype).view(B, 1, 1) * d_sine - 1).clamp(min=1.0)
    return -spread + 2.0 * spread * slot_idx / denom  # (B, L, d_sine)


class SpectralEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = getattr(cfg, "clip_encoder_mode", "spectral")
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        if self.mode == "spectral":
            # head emits 3·d_sine scalars per token: d_sine independent (A, f, φ)
            # triples that will sum into a d_sine-channel waveform.
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.GELU(),
                nn.Linear(cfg.d_model, 3 * cfg.d_sine),
            )
        else:
            self.head = None
        if self.mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(cfg.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    def _encode_hidden(self, tokens, pad_mask=None):
        # Returns (h, eff_mask) where h is (B, L_eff, d_model) post-trunk
        # hidden states and eff_mask is the matching pad mask (extended with a
        # False at position 0 in "cls" mode).
        cfg = self.cfg
        B, L = tokens.shape
        emb = self.token_emb(tokens)
        if self.mode == "cls":
            cls = self.cls_token.view(1, 1, -1).expand(B, -1, -1).to(emb.dtype)
            emb = torch.cat([cls, emb], dim=1)
            if pad_mask is not None:
                cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
                pad_mask = torch.cat([cls_mask, pad_mask], dim=1)
            L = L + 1
        pe = sinusoidal_pe(L, cfg.d_model, tokens.device, emb.dtype)
        x = emb + pe
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return h, pad_mask

    def hidden_states(self, tokens, pad_mask=None):
        """Public access to post-trunk hidden states for baseline pooling."""
        return self._encode_hidden(tokens, pad_mask)

    def forward(self, tokens, pad_mask=None):
        # tokens: (B, L) — L can be anything
        # pad_mask: (B, L) bool with True at PADDING positions (PyTorch convention).
        #   Threaded into the transformer as src_key_padding_mask AND used to
        #   zero A at padded slots so they contribute 0 to the synthesis sum.
        if self.head is None:
            raise RuntimeError(
                f"SpectralEncoder.forward called in mode={self.mode!r}; "
                f"use hidden_states(...) for baseline pooling."
            )
        cfg = self.cfg
        B, L = tokens.shape
        h, _ = self._encode_hidden(tokens, pad_mask)
        raw = self.head(h).view(B, L, 3, cfg.d_sine)
        raw_A, raw_f, raw_phi = raw.unbind(-2)  # each (B, L, d_sine)
        real_lengths = (~pad_mask).sum(dim=-1) if pad_mask is not None else None
        f_bias = f_init_bias(
            L, cfg.d_sine, cfg.f_bias_spread, tokens.device, raw_f.dtype, real_lengths=real_lengths
        )
        A = F.softplus(raw_A).clamp(max=cfg.A_max)
        f = cfg.f_min + (cfg.f_max - cfg.f_min) * torch.sigmoid(raw_f + f_bias)
        phi = 2 * math.pi * torch.sigmoid(raw_phi)
        if pad_mask is not None:
            keep = (~pad_mask).to(A.dtype).unsqueeze(-1)  # (B, L, 1)
            A = A * keep
        return A, f, phi


def synthesize(A, f, phi, n_samples: int, duration: float):
    """Multi-channel sum of sines.

    For each output channel j:
        s_j(t) = Σ_i A_{i,j} sin(2π f_{i,j} t + φ_{i,j})

    A, f, phi: (B, L, d_sine). Returns (B, n_samples, d_sine).
    """
    device = A.device
    t = torch.arange(n_samples, device=device, dtype=A.dtype) * (duration / n_samples)
    # broadcast (B, L, d_sine, 1) with (1, 1, 1, N)
    arg = 2 * math.pi * f.unsqueeze(-1) * t.view(1, 1, 1, -1) + phi.unsqueeze(-1)
    # sum over tokens (L axis) → (B, d_sine, N), then move N to seq dim
    signal = (A.unsqueeze(-1) * torch.sin(arg)).sum(dim=1)  # (B, d_sine, N)
    return signal.transpose(-1, -2).contiguous()  # (B, N, d_sine)


def freq_separation_loss(f, min_sep: float):
    """Hinge penalty on too-close pairs of predicted frequencies.

    f: (B, K). Returns a scalar averaged over off-diagonal pairs and batch.
    """
    diff = (f.unsqueeze(-1) - f.unsqueeze(-2)).abs()  # (B, K, K)
    K = f.size(-1)
    eye = torch.eye(K, dtype=torch.bool, device=f.device)
    penalty = torch.relu(min_sep - diff).masked_fill(eye, 0.0)
    return penalty.sum() / (f.size(0) * K * (K - 1))


class WaveformDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Linear-project each sample's d_sine channels into d_model.
        self.proj = nn.Linear(cfg.d_sine, cfg.d_model)
        # Learned positional embedding over the n_samples-long memory so the
        # decoder knows which sample is which. n_samples is fixed in cfg, so
        # this stays learned (only seq_len needed to become variable).
        self.mem_pos_emb = nn.Parameter(torch.zeros(1, cfg.n_samples, cfg.d_model))
        nn.init.normal_(self.mem_pos_emb, std=0.02)
        self.mem_norm = nn.LayerNorm(cfg.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(dec_layer, num_layers=cfg.decoder_layers)

    def forward(self, signal, L: int):
        # signal: (B, N, d_sine); L = number of output token slots requested
        cfg = self.cfg
        B = signal.size(0)
        memory = self.mem_norm(self.proj(signal) + self.mem_pos_emb)
        # Sinusoidal positional queries — one per output slot. Variable in L.
        queries = sinusoidal_pe(L, cfg.d_model, signal.device, signal.dtype).expand(
            B, -1, -1
        )
        return self.transformer(queries, memory)  # (B, L, d_model)


class SpectralAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectralEncoder(cfg)
        # Waveform decoder is only meaningful in spectral mode (it cross-attends
        # over the synthesized waveform). Baseline modes skip it entirely.
        if getattr(cfg, "clip_encoder_mode", "spectral") == "spectral":
            self.decoder = WaveformDecoder(cfg)
        else:
            self.decoder = None

    def forward(self, tokens):
        cfg = self.cfg
        L = tokens.size(1)
        A, f, phi = self.encoder(tokens)  # each (B, L, d_sine)
        signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)  # (B, N, d_sine)
        decoded_emb = self.decoder(signal, L)  # (B, L, d_model)
        # tied output projection
        logits = decoded_emb @ self.encoder.token_emb.weight.T  # (B, L, V)
        # Frequency-separation aux: flatten (L, d_sine) so the penalty pushes
        # ALL waves apart, both across tokens and across channels within a
        # token. Without this, distinct channels could collapse to identical
        # frequencies and waste capacity.
        min_sep = cfg.freq_sep_min_bins / cfg.duration
        aux = freq_separation_loss(f.flatten(1, 2), min_sep)
        return logits, tokens, aux
