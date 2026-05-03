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
            nn.Linear(cfg.d_model, cfg.d_model/2),
            nn.GELU(),
            nn.Linear(cfg.d_model/2, cfg.d_model/4),
            nn.GELU(),
            nn.Linear(cfg.d_model/4, 3)
        )

    def forward(self, tokens):
        # tokens: (B, L)
        x = self.token_emb(tokens) + self.pos_emb
        h = self.encoder(x)  # no causal mask, no padding mask
        raw = self.head(h)  # (B, L, 3)
        raw_A, raw_f, raw_phi = raw.unbind(-1)
        cfg = self.cfg
        A = F.softplus(raw_A)
        f = cfg.f_min + (cfg.f_max - cfg.f_min) * torch.sigmoid(raw_f)
        phi = 2 * math.pi * torch.sigmoid(raw_phi)
        # sort by frequency along L for canonical ordering
        f_sorted, perm = torch.sort(f, dim=-1, stable=True)
        A_sorted = torch.gather(A, -1, perm)
        phi_sorted = torch.gather(phi, -1, perm)
        return A_sorted, f_sorted, phi_sorted, perm


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
    """Window+rFFT, take top-K bins (excluding DC), parabolic-interp the
    sub-bin frequency, recover (Â, f̂, φ̂).

    signal: (B, N). Returns A_hat, f_hat, phi_hat, each (B, n_peaks).
    Outputs are sorted by ascending f̂.
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
    delta = torch.where(denom.abs() > 1e-8, 0.5 * (a - c) / denom, torch.zeros_like(denom))
    delta = delta.clamp(-0.5, 0.5)
    # zero delta at boundary bins
    boundary = (topk_idx_d == 0) | (topk_idx_d == M - 1)
    delta = torch.where(boundary, torch.zeros_like(delta), delta)

    f_hat = (topk_idx_d.to(signal.dtype) + delta) / duration
    # amplitude: peak-amp of windowed sine ≈ |X[k]| * 2 / window.sum()
    A_hat = torch.gather(mag, -1, topk_idx_d) * (2.0 / win_sum)
    # phase: angle of X at peak; sin convention shifts cos-phase by -π/2
    real = torch.gather(X.real, -1, topk_idx_d)
    imag = torch.gather(X.imag, -1, topk_idx_d)
    phi_hat = torch.atan2(imag, real) + math.pi / 2
    phi_hat = (phi_hat + 2 * math.pi) % (2 * math.pi)

    # sort by f̂
    f_hat, sort_idx = torch.sort(f_hat, dim=-1, stable=True)
    A_hat = torch.gather(A_hat, -1, sort_idx)
    phi_hat = torch.gather(phi_hat, -1, sort_idx)
    return A_hat, f_hat, phi_hat


class ParamDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = 4  # Â, f̂_norm, sin φ̂, cos φ̂
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.d_model),
        )

    def forward(self, A_hat, f_hat, phi_hat):
        cfg = self.cfg
        f_norm = (f_hat - cfg.f_min) / (cfg.f_max - cfg.f_min)
        feats = torch.stack([A_hat, f_norm, torch.sin(phi_hat), torch.cos(phi_hat)], dim=-1)
        return self.net(feats)  # (B, L, d_model)


class SpectralAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectralEncoder(cfg)
        self.decoder = ParamDecoder(cfg)

    def forward(self, tokens):
        cfg = self.cfg
        A, f, phi, perm = self.encoder(tokens)
        signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
        A_hat, f_hat, phi_hat = fft_peaks(signal, cfg.seq_len, cfg.duration)
        decoded_emb = self.decoder(A_hat, f_hat, phi_hat)  # (B, L, d_model)
        # tied output projection
        logits = decoded_emb @ self.encoder.token_emb.weight.T  # (B, L, V)
        # reorder targets to match the encoder-side sort by predicted f
        sorted_tokens = torch.gather(tokens, -1, perm)
        return logits, sorted_tokens
