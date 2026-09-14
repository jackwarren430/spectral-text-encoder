"""Configuration dedicated to the spectral VAE training pipeline."""

from dataclasses import asdict, dataclass
import json
import os


@dataclass
class VAEConfig:
    # Data.  max_length includes the explicit EOS token.
    tokenizer_name: str = "gpt2"
    vocab_size: int = 50257
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    vae_max_length: int = 128
    vae_val_frac: float = 0.05
    vae_dataset_specs: tuple = (
        ("sentence-transformers/all-nli", "pair"),
        ("sentence-transformers/quora-duplicates", "pair"),
        ("sentence-transformers/altlex", ""),
        ("sentence-transformers/stackexchange-duplicates", "title-title-pair"),
        ("sentence-transformers/coco-captions", "pair"),
        ("sentence-transformers/sentence-compression", "pair"),
    )

    # Shared contextual text encoder.
    d_model: int = 512
    encoder_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1

    # Spectral latent.  The first global_bands are reserved for the global
    # posterior; every remaining band is reserved for position-coded token
    # latents.  There is no encoder-to-decoder path around the scalar waveform.
    n_samples: int = 2048
    duration: float = 1.0
    f_min: float = 1.0
    f_max: float = 960.0
    n_bands: int = 12
    global_bands: int = 4
    global_latent_per_band: int = 4
    token_latent_dim: int = 4
    latent_logvar_min: float = -10.0
    latent_logvar_max: float = 6.0
    token_sum_normalize: bool = True
    spectral_basis_seed: int = 1729

    # Fourier-aware, non-autoregressive decoder.
    spectral_conv_channels: int = 128
    spectral_conv_layers: int = 2
    spectral_patch_size: int = 4
    decoder_layers: int = 4

    # ELBO optimization.  Every objective component is normalized by the
    # number of real target tokens, so reconstruction and KL scaling do not
    # silently change with batch padding.
    vae_batch_size: int = 128
    vae_grad_accum_steps: int = 1
    vae_lr: float = 1.5e-4
    weight_decay: float = 0.05
    vae_warmup_steps: int = 1500
    vae_max_steps: int = 50000
    grad_clip: float = 1.2
    beta_global: float = 1.0
    beta_token: float = 0.1
    kl_reconstruction_warmup_steps: int = 1000
    kl_anneal_steps: int = 10000
    global_free_bits: float = 0.05
    token_free_bits: float = 0.05
    active_unit_variance: float = 0.01

    # Logging, validation, and checkpointing.
    vae_log_every: int = 100
    vae_val_every: int = 500
    vae_val_batches: int = 50
    vae_ckpt_every: int = 1000
    vae_ckpt_dir: str = "all-training/spectral-vae-v1"
    vae_validate_sample_reconstruction: bool = True
    vae_validate_signal_ablations: bool = True
    vae_prior_samples: int = 16

    # Runtime.
    device: str = "auto"
    precision: str = "bf16"
    cuda_tf32: bool = True
    fused_optimizer: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    seed: int = 0

    def __post_init__(self):
        # JSON snapshots turn tuples into lists; normalize them on resume.
        self.vae_dataset_specs = tuple(tuple(spec) for spec in self.vae_dataset_specs)
        if self.d_model % 2 or self.d_model % self.n_heads:
            raise ValueError("d_model must be even and divisible by n_heads")
        if self.vae_max_length < 2:
            raise ValueError("vae_max_length must leave room for text and EOS")
        if self.n_samples < 8:
            raise ValueError("n_samples must be at least 8")
        nyquist = self.n_samples / (2.0 * self.duration)
        if not 0.0 < self.f_min < self.f_max < nyquist:
            raise ValueError(
                f"require 0 < f_min < f_max < Nyquist ({nyquist:g}); "
                f"got {self.f_min:g}, {self.f_max:g}"
            )
        if not 0 < self.global_bands < self.n_bands:
            raise ValueError("global_bands must reserve some, but not all, bands")
        if min(self.global_latent_per_band, self.token_latent_dim) < 1:
            raise ValueError("latent dimensions must be positive")
        if self.spectral_patch_size < 1 or self.spectral_conv_layers < 1:
            raise ValueError("spectral decoder depth and patch size must be positive")
        if self.vae_batch_size < 1 or self.vae_grad_accum_steps < 1:
            raise ValueError("VAE batch and accumulation sizes must be positive")
        if min(self.beta_global, self.beta_token) < 0:
            raise ValueError("KL weights must be non-negative")
        if min(self.global_free_bits, self.token_free_bits) < 0:
            raise ValueError("free-bits thresholds must be non-negative")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")


def save_vae_config(cfg: VAEConfig, run_dir: str) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_vae_config(run_dir: str) -> VAEConfig:
    with open(os.path.join(run_dir, "config.json")) as f:
        return VAEConfig(**json.load(f))
