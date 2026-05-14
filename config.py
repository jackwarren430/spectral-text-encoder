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
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    decoder_layers: int = 8
    # per-token sine-wave channel count: each of the L encoder slots emits
    # d_sine independent (A, f, φ) triples. The decoder consumes the summed
    # multi-channel waveform directly (no FFT round-trip).
    d_sine: int = 6

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
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    max_steps: int = 100000
    grad_clip: float = 1.2
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 50
    ckpt_every: int = 1000
    ckpt_dir: str = "all-training/frozen-encoder/"
    # Freeze the encoder (and its tied token_emb output projection) during AE
    # training. Intended use: load a CLIP-trained checkpoint via --init-from
    # and train only the decoder to reconstruct from the frozen waveforms.
    freeze_encoder: bool = True

    # CLIP-style contrastive training (train_clip.py)
    clip_dataset_name: str = "sentence-transformers/all-nli"
    clip_dataset_config: str = "pair"
    clip_max_len: int = 128  # truncate longer sentences (GPT-2 BPE tokens)
    clip_val_frac: float = 0.05  # last 5% of train held out as validation
    clip_batch_size: int = 512
    # Gradient accumulation: each optimizer step backpropagates the average over
    # this many mini-batches. Smooths gradient direction estimates without
    # raising peak memory. Note: this does NOT give more in-batch negatives —
    # each mini-batch still computes its loss against its own (B-1) negatives.
    clip_grad_accum_steps: int = 1
    clip_lr: float = 1.5e-4
    clip_warmup_steps: int = 3000
    clip_max_steps: int = 100000
    clip_logit_scale_init: float = 2.6593 # ln(1/0.07) — CLIP default
    clip_logit_scale_max: float = 4.6052  # ln(100) — clamp ceiling per CLIP
    clip_log_every: int = 100
    clip_val_every: int = 1000
    clip_val_batches: int = 50
    clip_ckpt_every: int = 2000
    clip_ckpt_dir: str = "all-training/runpod"
    # Gradient caching (Gao et al. 2021). When set and < clip_batch_size, the
    # contrastive loss is computed across the full clip_batch_size of negatives
    # while only chunk_size examples are forwarded with grad at a time. Lets
    # you raise clip_batch_size (more negatives) without raising peak memory.
    # None disables (single forward pass, current behavior).
    clip_cache_chunk_size: int = 64
    # Multi-source contrastive mix. Each entry is (hf_dataset_name, config_name);
    # use "" for datasets without a config. When non-empty, this overrides the
    # legacy single-source clip_dataset_name / clip_dataset_config. Empty tuple
    # = use the legacy single-source path (clip_dataset_name + clip_dataset_config).
    clip_dataset_specs: tuple = (                                                                                                                      
        ("sentence-transformers/all-nli", "pair"),                                                                                                     
        ("sentence-transformers/quora-duplicates", "pair"),                                                                                            
        ("sentence-transformers/altlex", ""),                                                                                                          
    )
    # Embedding for the contrastive loss. "time": flatten the synthesized
    # waveform directly (current behavior). "spectral": take |rfft(signal)| per
    # channel before flatten + L2-normalize — phase-invariant; dimensionality
    # drops to (N//2+1) * d_sine.
    clip_embedding_type: str = "time"
    # Per-channel InfoNCE auxiliary loss. For each of d_sine channels, compute
    # a contrastive loss using just that channel's slice of the signal and
    # average across channels. Pushes the encoder to keep channels distinct,
    # directly attacking the d_sine-collapse failure mode. 0 disables.
    clip_per_channel_lambda: float = 0.1
    # Reconstruction auxiliary loss. Runs the decoder on the synthesized
    # waveform and computes CE against the input tokens (AE-mode objective) on
    # both anchor and positive sides. Decoder params join the optimizer when
    # this is > 0. 0 disables.
    clip_recon_lambda: float = 0.0

    # runtime
    device: str = "cuda"
    num_workers: int = 16
    seed: int = 0
