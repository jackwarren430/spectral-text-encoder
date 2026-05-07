from dataclasses import dataclass


@dataclass
class Config:
    # data
    seq_len: int = 16
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
    decoder_layers: int = 6
    # per-token sine-wave channel count: each of the L encoder slots emits
    # d_sine independent (A, f, φ) triples. The decoder consumes the summed
    # multi-channel waveform directly (no FFT round-trip).
    d_sine: int = 2

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
    freq_sep_lambda: float = 0.05

    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 20000
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 50
    ckpt_every: int = 2000
    ckpt_dir: str = "all-training/checkpoints"

    # CLIP-style contrastive training (train_clip.py)
    clip_dataset_name: str = "sentence-transformers/all-nli"
    clip_dataset_config: str = "pair"
    clip_max_len: int = 128  # truncate longer sentences (GPT-2 BPE tokens)
    clip_val_frac: float = 0.05  # last 5% of train held out as validation
    clip_batch_size: int = 64
    clip_lr: float = 3e-4
    clip_warmup_steps: int = 1000
    clip_max_steps: int = 20000
    clip_logit_scale_init: float = 2.6593  # ln(1/0.07) — CLIP default
    clip_logit_scale_max: float = 4.6052  # ln(100) — clamp ceiling per CLIP
    clip_log_every: int = 50
    clip_val_every: int = 1000
    clip_val_batches: int = 50
    clip_ckpt_every: int = 2000
    clip_ckpt_dir: str = "all-training/clip_checkpoints"

    # runtime
    device: str = "mps"
    num_workers: int = 2
    seed: int = 0
