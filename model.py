import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
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
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 3)
        )
        # Per-position pre-sigmoid bias on the f channel. Linspace gives an
        # initial frequency spread across [f_min, f_max] so peaks don't all
        # collapse onto a single bin at step 0.
        self.f_pos_bias = nn.Parameter(
            torch.linspace(-cfg.f_bias_spread, cfg.f_bias_spread, cfg.seq_len).view(1, -1)
        )

    def forward(self, tokens):
        # tokens: (B, L)
        x = self.token_emb(tokens) + self.pos_emb
        h = self.encoder(x)  # no causal mask, no padding mask
        raw = self.head(h)  # (B, L, 3)
        raw_A, raw_f, raw_phi = raw.unbind(-1)
        cfg = self.cfg
        A = F.softplus(raw_A).clamp(max=cfg.A_max)
        f = cfg.f_min + (cfg.f_max - cfg.f_min) * torch.sigmoid(raw_f + self.f_pos_bias)
        phi = 2 * math.pi * torch.sigmoid(raw_phi)
        return A, f, phi


def synthesize(A, f, phi, n_samples: int, duration: float):
    """Sum of sines: s(t) = Σ_i A_i sin(2π f_i t + φ_i).

    A, f, phi: (B, L). Returns (B, n_samples).
    """
    device = A.device
    t = torch.arange(n_samples, device=device, dtype=A.dtype) * (duration / n_samples)
    # (B, L, 1) broadcast with (1, 1, N)
    arg = 2 * math.pi * f.unsqueeze(-1) * t.view(1, 1, -1) + phi.unsqueeze(-1)
    signal = (A.unsqueeze(-1) * torch.sin(arg)).sum(dim=1)
    return signal


def fft_peaks(signal, n_peaks: int, duration: float, eps: float = 1e-10):
    """Window+rFFT, take top-K local-maxima bins (excluding DC), parabolic-interp
    the sub-bin frequency, recover (Â, f̂, φ̂).

    signal: (B, N). Returns A_hat, f_hat, phi_hat, each (B, n_peaks).
    Outputs are ordered by descending magnitude (largest peak first).
    """
    B, N = signal.shape
    device = signal.device
    window = torch.hann_window(N, periodic=False, device=device, dtype=signal.dtype)
    win_sum = window.sum()
    X = torch.fft.rfft(signal * window, n=N)
    mag = X.abs()  # (B, M) where M = N//2 + 1
    M = mag.size(-1)
    # local-maxima mask (interior bins only): mag[k] > mag[k-1] and mag[k] > mag[k+1]
    interior = mag[:, 1:-1]
    is_peak = (interior > mag[:, :-2]) & (interior > mag[:, 2:])
    mag_for_pick = torch.zeros_like(mag)
    mag_for_pick[:, 1:-1] = torch.where(is_peak, interior, torch.zeros_like(interior))
    # top-K among local maxima
    topk_idx = mag_for_pick.topk(n_peaks, dim=-1).indices
    topk_idx_d = topk_idx.detach()  # selection is non-differentiable
    left_idx = (topk_idx_d - 1).clamp(min=0)
    right_idx = (topk_idx_d + 1).clamp(max=M - 1)
    log_mag = torch.log(mag + eps)
    a = torch.gather(log_mag, -1, left_idx)
    b = torch.gather(log_mag, -1, topk_idx_d)
    c = torch.gather(log_mag, -1, right_idx)
    denom = a - 2 * b + c
    # Gradient-safe division: sanitize the denominator BEFORE dividing so
    # autograd never propagates through a 1/0 in the unselected branch.
    denom_ok = denom.abs() > 1e-8
    safe_denom = torch.where(denom_ok, denom, torch.ones_like(denom))
    delta = torch.where(denom_ok, 0.5 * (a - c) / safe_denom, torch.zeros_like(denom))
    delta = delta.clamp(-0.5, 0.5)
    # zero delta at boundary bins
    boundary = (topk_idx_d == 0) | (topk_idx_d == M - 1)
    delta = torch.where(boundary, torch.zeros_like(delta), delta)

    f_hat = (topk_idx_d.to(signal.dtype) + delta) / duration
    # amplitude: peak-amp of windowed sine ≈ |X[k]| * 2 / window.sum()
    A_hat = torch.gather(mag, -1, topk_idx_d) * (2.0 / win_sum)
    # phase: angle of X at peak; sin convention shifts cos-phase by π/2.
    # Sub-bin correction: a sine at bin k+δ produces a linear phase ramp of
    # roughly π·δ at the integer bin k, so subtract it back out.
    real = torch.gather(X.real, -1, topk_idx_d)
    imag = torch.gather(X.imag, -1, topk_idx_d)
    phi_hat = torch.atan2(imag, real) + math.pi / 2 - math.pi * delta
    phi_hat = (phi_hat + 2 * math.pi) % (2 * math.pi)

    # topk already returns indices ordered by descending magnitude, so the
    # outputs are magnitude-sorted (largest peak first) without further work.
    return A_hat, f_hat, phi_hat


def fourier_features(x, K: int):
    """NeRF-style multi-scale sin/cos expansion of a low-dim input.

    x: (..., D). Returns (..., D * 2 * K) by stacking [sin(2^k π x), cos(2^k π x)]
    for k = 0..K-1.
    """
    bands = (2.0 ** torch.arange(K, device=x.device, dtype=x.dtype)) * math.pi
    args = x.unsqueeze(-1) * bands  # (..., D, K)
    feats = torch.cat([args.sin(), args.cos()], dim=-1)  # (..., D, 2K)
    return feats.flatten(-2)  # (..., D*2K)


def freq_separation_loss(f, min_sep: float):
    """Hinge penalty on too-close pairs of predicted frequencies.

    f: (B, L). Returns a scalar averaged over off-diagonal pairs and batch.
    """
    diff = (f.unsqueeze(-1) - f.unsqueeze(-2)).abs()  # (B, L, L)
    L = f.size(-1)
    eye = torch.eye(L, dtype=torch.bool, device=f.device)
    penalty = torch.relu(min_sep - diff).masked_fill(eye, 0.0)
    return penalty.sum() / (f.size(0) * L * (L - 1))


class ParamDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 4 base scalars (Â, f̂_norm, sin φ̂, cos φ̂) lifted via Fourier features
        in_dim = 4 * 2 * cfg.fourier_K

        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.d_model),
        )

        # Learned positional queries, one per output slot. Slot i is meant to
        # reconstruct the i-th original token; the transformer cross-attends
        # to the magnitude-sorted peak features to pull whatever it needs.
        self.queries = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.queries, std=0.02)

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

    def forward(self, A_hat, f_hat, phi_hat):
        cfg = self.cfg
        f_norm = (f_hat - cfg.f_min) / (cfg.f_max - cfg.f_min)
        base = torch.stack([A_hat, f_norm, torch.sin(phi_hat), torch.cos(phi_hat)], dim=-1)
        feats = fourier_features(base, cfg.fourier_K)
        memory = self.net(feats)  # (B, L, d_model), magnitude-sorted peak features
        B = memory.size(0)
        queries = self.queries.expand(B, -1, -1)
        return self.transformer(queries, memory)  # (B, L, d_model)


class SpectralAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectralEncoder(cfg)
        self.decoder = ParamDecoder(cfg)

    def forward(self, tokens):
        cfg = self.cfg
        A, f, phi = self.encoder(tokens)
        signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
        A_hat, f_hat, phi_hat = fft_peaks(signal, cfg.seq_len, cfg.duration)
        decoded_emb = self.decoder(A_hat, f_hat, phi_hat)  # (B, L, d_model)
        # tied output projection
        logits = decoded_emb @ self.encoder.token_emb.weight.T  # (B, L, V)
        # frequency-separation auxiliary penalty (in Hz; min_sep_bins · Δf)
        min_sep = cfg.freq_sep_min_bins / cfg.duration
        aux = freq_separation_loss(f, min_sep)
        return logits, tokens, aux
