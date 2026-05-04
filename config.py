from dataclasses import dataclass


@dataclass
class Config:
    # data
    seq_len: int = 64
    tokenizer_name: str = "gpt2"
    vocab_size: int = 50257
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    # model
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ffn_dim: int = 2048
    dropout: float = 0.0
    decoder_hidden: int = 256
    decoder_layers: int = 2
    fourier_K: int = 6  # bands for decoder Fourier-feature encoding

    # signal / FFT
    n_samples: int = 2048
    duration: float = 1.0
    f_min: float = 1.0
    f_max: float = 960.0  # Nyquist = n_samples / (2*duration) = 1024; leave margin

    # encoder f-head init: per-position bias spread (in pre-sigmoid space)
    f_bias_spread: float = 3.0

    # amplitude cap (softplus(raw_A).clamp(max=...)) to prevent runaway
    A_max: float = 10.0

    # frequency-separation auxiliary loss
    freq_sep_min_bins: float = 4.0  # main-lobe width for Hann ≈ 4 bins
    freq_sep_lambda: float = 1e-3

    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1500
    max_steps: int = 20000
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 50
    ckpt_every: int = 2000
    ckpt_dir: str = "checkpoints"

    # runtime
    device: str = "mps"
    num_workers: int = 2
    seed: int = 0
