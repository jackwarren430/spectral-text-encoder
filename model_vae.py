"""Generative spectral VAE with a single scalar-waveform bottleneck."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import sinusoidal_pe


@dataclass
class SpectralVAEOutput:
    logits: torch.Tensor
    waveform: torch.Tensor
    spectrum: torch.Tensor
    global_spectrum: torch.Tensor
    token_spectrum: torch.Tensor
    mu_global: torch.Tensor
    logvar_global: torch.Tensor
    mu_token: torch.Tensor
    logvar_token: torch.Tensor
    attention: torch.Tensor


class VAETextEncoder(nn.Module):
    """Contextual encoder plus global and per-token Gaussian posteriors."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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
        # Keep this attribute name aligned with SpectralEncoder so an existing
        # CLIP checkpoint can initialize token_emb + encoder without remapping.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.encoder_layers)
        self.global_query = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(self.global_query, std=0.02)
        global_size = cfg.global_bands * cfg.global_latent_per_band
        self.global_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2 * global_size),
        )
        self.token_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.token_latent_dim),
        )

    def forward(self, tokens, pad_mask):
        if tokens.size(1) > self.cfg.vae_max_length:
            raise ValueError(
                f"input length {tokens.size(1)} exceeds VAE maximum "
                f"{self.cfg.vae_max_length}"
            )
        emb = self.token_emb(tokens)
        emb = emb + sinusoidal_pe(
            tokens.size(1), self.cfg.d_model, tokens.device, emb.dtype
        )
        hidden = self.encoder(emb, src_key_padding_mask=pad_mask)

        # Posterior statistics remain FP32 under CUDA BF16.  This avoids coarse
        # log-variance quantization and gives the FFT path a stable input.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.float()
            scores = torch.einsum("bld,d->bl", hidden, self.global_query.float())
            scores = scores / math.sqrt(self.cfg.d_model)
            scores = scores.masked_fill(pad_mask, float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            pooled = torch.einsum("bl,bld->bd", attention, hidden)

            global_stats = self.global_head(pooled).view(
                tokens.size(0), self.cfg.global_bands,
                2 * self.cfg.global_latent_per_band,
            )
            mu_global, logvar_global = global_stats.split(
                self.cfg.global_latent_per_band, dim=-1
            )
            token_stats = self.token_head(hidden)
            mu_token, logvar_token = token_stats.split(
                self.cfg.token_latent_dim, dim=-1
            )
            logvar_global = logvar_global.clamp(
                self.cfg.latent_logvar_min, self.cfg.latent_logvar_max
            )
            logvar_token = logvar_token.clamp(
                self.cfg.latent_logvar_min, self.cfg.latent_logvar_max
            )
        return (
            mu_global, logvar_global, mu_token, logvar_token, attention
        )


class SpectralLatentLayout(nn.Module):
    """Map global and position-coded token latents into reserved FFT bands."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_bins = cfg.n_samples // 2 + 1
        frequencies = torch.arange(n_bins, dtype=torch.float32) / cfg.duration
        active = (frequencies >= cfg.f_min) & (frequencies <= cfg.f_max)
        active_indices = active.nonzero(as_tuple=False).flatten()
        if active_indices.numel() < cfg.n_bands:
            raise ValueError(
                f"only {active_indices.numel()} usable FFT bins for {cfg.n_bands} bands"
            )

        band_masks = torch.zeros(cfg.n_bands, n_bins, dtype=torch.bool)
        # tensor_split permits non-divisible layouts while keeping every band
        # contiguous and within the configured frequency range.
        for band, indices in enumerate(torch.tensor_split(active_indices, cfg.n_bands)):
            band_masks[band, indices] = True
        global_mask = band_masks[: cfg.global_bands].any(dim=0)
        token_mask = band_masks[cfg.global_bands :].any(dim=0)
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("band_masks", band_masks)
        self.register_buffer("global_mask", global_mask)
        self.register_buffer("token_mask", token_mask)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.spectral_basis_seed)
        global_basis = torch.randn(
            cfg.global_bands,
            cfg.global_latent_per_band,
            n_bins,
            2,
            generator=generator,
        )
        global_basis *= band_masks[: cfg.global_bands, None, :, None]
        global_basis = self._normalize_basis(global_basis)
        self.global_basis = nn.Parameter(global_basis)

        token_basis = torch.randn(
            cfg.vae_max_length,
            cfg.token_latent_dim,
            n_bins,
            2,
            generator=generator,
        )
        token_basis *= token_mask[None, None, :, None]
        token_basis = self._normalize_basis(token_basis)
        # Slot codes are deliberately fixed: their stable identity is what
        # makes per-token contributions recoverable after spectral summation.
        self.register_buffer("token_basis", token_basis)

    @staticmethod
    def _normalize_basis(basis):
        norm = basis.square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
        return basis / norm

    def normalized_global_basis(self):
        mask = self.band_masks[: self.cfg.global_bands, None, :, None]
        return self._normalize_basis(self.global_basis * mask)

    @staticmethod
    def _as_complex(coefficients):
        return torch.complex(coefficients[..., 0], coefficients[..., 1])

    def forward(self, z_global, z_token, pad_mask):
        # torch.complex has no BF16 constructor, and BF16 would unnecessarily
        # quantize the spectral coefficients. Keep basis projection and complex
        # construction in FP32 even when the surrounding transformer uses BF16.
        with torch.autocast(device_type=z_global.device.type, enabled=False):
            z_global = z_global.float()
            z_token = z_token.float()
            global_coefficients = torch.einsum(
                "bgd,gdkc->bkc", z_global, self.normalized_global_basis().float()
            )
            keep = (~pad_mask).to(z_token.dtype).unsqueeze(-1)
            masked_tokens = z_token * keep
            token_coefficients = torch.einsum(
                "bld,ldkc->bkc",
                masked_tokens,
                self.token_basis[: z_token.size(1)].float(),
            )
            if self.cfg.token_sum_normalize:
                lengths = keep.sum(dim=1).clamp(min=1.0).sqrt()
                token_coefficients = token_coefficients / lengths.unsqueeze(-1)
            global_spectrum = self._as_complex(global_coefficients)
            token_spectrum = self._as_complex(token_coefficients)
        spectrum = global_spectrum + token_spectrum
        return spectrum, global_spectrum, token_spectrum

    def band_energies(self, spectrum, log: bool = True):
        values = []
        power = spectrum.abs().square()
        for mask in self.band_masks:
            energy = power[:, mask].mean(dim=-1)
            values.append(torch.log(energy + 1e-8) if log else energy)
        return torch.stack(values, dim=-1)

    def edit_band_gain(self, waveform, band: int, gain: float):
        if not 0 <= band < self.cfg.n_bands:
            raise ValueError(f"band must be in [0, {self.cfg.n_bands}), got {band}")
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.fft.rfft(
                waveform.squeeze(-1).float(), n=self.cfg.n_samples, dim=-1, norm="ortho"
            )
            edited = spectrum.clone()
            edited[:, self.band_masks[band]] *= gain
            return torch.fft.irfft(
                edited, n=self.cfg.n_samples, dim=-1, norm="ortho"
            ).unsqueeze(-1)


class FourierWaveformDecoder(nn.Module):
    """Complex-rFFT feature extractor and non-autoregressive token decoder."""

    def __init__(self, cfg, band_masks):
        super().__init__()
        self.cfg = cfg
        n_features = 4 + cfg.n_bands + 2
        layers = []
        in_channels = n_features
        for _ in range(cfg.spectral_conv_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels, cfg.spectral_conv_channels,
                        kernel_size=5, padding=2,
                    ),
                    nn.GELU(),
                ]
            )
            in_channels = cfg.spectral_conv_channels
        self.local_features = nn.Sequential(*layers)
        self.patch = nn.Conv1d(
            cfg.spectral_conv_channels,
            cfg.d_model,
            kernel_size=cfg.spectral_patch_size,
            stride=cfg.spectral_patch_size,
        )
        self.memory_norm = nn.LayerNorm(cfg.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.decoder_layers
        )
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.register_buffer("band_features", band_masks.T.to(torch.float32))
        self.register_buffer(
            "normalized_frequency",
            torch.linspace(0.0, 1.0, cfg.n_samples // 2 + 1),
        )

    def spectral_memory(self, waveform):
        if waveform.ndim != 3 or waveform.size(-1) != 1:
            raise ValueError(
                f"decoder requires a scalar waveform (B, N, 1), got {tuple(waveform.shape)}"
            )
        if waveform.size(1) != self.cfg.n_samples:
            raise ValueError(
                f"decoder expected {self.cfg.n_samples} samples, got {waveform.size(1)}"
            )
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.fft.rfft(
                waveform.squeeze(-1).float(), dim=-1, norm="ortho"
            )
            magnitude = spectrum.abs()
            batch = waveform.size(0)
            frequency = self.normalized_frequency.expand(batch, -1)
            bands = self.band_features.unsqueeze(0).expand(batch, -1, -1)
            global_flag = bands[:, :, : self.cfg.global_bands].sum(-1, keepdim=True)
            token_flag = bands[:, :, self.cfg.global_bands :].sum(-1, keepdim=True)
            features = torch.cat(
                [
                    spectrum.real.unsqueeze(-1),
                    spectrum.imag.unsqueeze(-1),
                    torch.log1p(magnitude).unsqueeze(-1),
                    frequency.unsqueeze(-1),
                    bands,
                    global_flag,
                    token_flag,
                ],
                dim=-1,
            )
        memory = self.patch(self.local_features(features.transpose(1, 2)))
        memory = memory.transpose(1, 2)
        memory = memory + sinusoidal_pe(
            memory.size(1), self.cfg.d_model, memory.device, memory.dtype
        )
        return self.memory_norm(memory)

    def forward(self, waveform):
        memory = self.spectral_memory(waveform)
        queries = sinusoidal_pe(
            self.cfg.vae_max_length,
            self.cfg.d_model,
            waveform.device,
            memory.dtype,
        ).expand(waveform.size(0), -1, -1)
        decoded = self.transformer(queries, memory)
        return self.output_norm(decoded)


class SpectralVAE(nn.Module):
    """Text -> stochastic complex spectrum -> waveform -> text."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = VAETextEncoder(cfg)
        self.latent_layout = SpectralLatentLayout(cfg)
        self.decoder = FourierWaveformDecoder(cfg, self.latent_layout.band_masks)

    @staticmethod
    def reparameterize(mu, logvar, sample: bool):
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def posterior(self, tokens, pad_mask, sample: bool = True):
        stats = self.encoder(tokens, pad_mask)
        mu_global, logvar_global, mu_token, logvar_token, attention = stats
        z_global = self.reparameterize(mu_global, logvar_global, sample)
        z_token = self.reparameterize(mu_token, logvar_token, sample)
        return (
            z_global, z_token, mu_global, logvar_global,
            mu_token, logvar_token, attention,
        )

    def spectrum_to_waveform(self, spectrum):
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            waveform = torch.fft.irfft(
                spectrum.to(torch.complex64),
                n=self.cfg.n_samples,
                dim=-1,
                norm="ortho",
            )
        return waveform.unsqueeze(-1)

    def encode_waveform(self, tokens, pad_mask, sample: bool = False):
        z_global, z_token, *_rest = self.posterior(tokens, pad_mask, sample=sample)
        spectrum, _, _ = self.latent_layout(z_global, z_token, pad_mask)
        return self.spectrum_to_waveform(spectrum)

    def decode_waveform(self, waveform):
        decoded = self.decoder(waveform)
        # GPT-2 embeddings initialize with unit-scale elements. LayerNorm plus
        # sqrt(d_model) scaling keeps tied-projection logits near unit scale.
        return (decoded @ self.encoder.token_emb.weight.T) / math.sqrt(self.cfg.d_model)

    def forward(self, tokens, pad_mask, sample: bool = True):
        (
            z_global, z_token, mu_global, logvar_global,
            mu_token, logvar_token, attention,
        ) = self.posterior(tokens, pad_mask, sample=sample)
        spectrum, global_spectrum, token_spectrum = self.latent_layout(
            z_global, z_token, pad_mask
        )
        waveform = self.spectrum_to_waveform(spectrum)
        logits = self.decode_waveform(waveform)
        return SpectralVAEOutput(
            logits=logits,
            waveform=waveform,
            spectrum=spectrum,
            global_spectrum=global_spectrum,
            token_spectrum=token_spectrum,
            mu_global=mu_global,
            logvar_global=logvar_global,
            mu_token=mu_token,
            logvar_token=logvar_token,
            attention=attention,
        )

    def prior_waveform(self, batch_size: int, device=None, token_mask=None):
        device = device or self.encoder.token_emb.weight.device
        z_global = torch.randn(
            batch_size,
            self.cfg.global_bands,
            self.cfg.global_latent_per_band,
            device=device,
        )
        z_token = torch.randn(
            batch_size,
            self.cfg.vae_max_length,
            self.cfg.token_latent_dim,
            device=device,
        )
        if token_mask is None:
            token_mask = torch.zeros(
                batch_size, self.cfg.vae_max_length, dtype=torch.bool, device=device
            )
        spectrum, _, _ = self.latent_layout(z_global, z_token, token_mask)
        return self.spectrum_to_waveform(spectrum)

    def edit_band_gain(self, waveform, band: int, gain: float):
        return self.latent_layout.edit_band_gain(waveform, band, gain)

    @staticmethod
    def interpolate_waveforms(first, second, alpha: float):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        return torch.lerp(first, second, alpha)
