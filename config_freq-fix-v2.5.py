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
    # Sine parameterization per token slot:
    #   "independent" (default): each slot emits d_sine independent (A, f, φ)
    #     triples — d_sine separate single-channel waves.
    #   "shared": each slot emits a d_sine-dim amplitude vector but a SINGLE
    #     frequency ω and phase φ shared across all channels — a true
    #     multi-dimensional sine f(t) = A·sin(ωt + φ), A ∈ R^d_sine. The head
    #     emits d_sine + 2 scalars per slot instead of 3·d_sine. Everything
    #     downstream (synthesize, decoder, embeddings) is unchanged because f
    #     and φ are broadcast to (B, L, d_sine) before synthesis.
    sine_param_mode: str = "independent"

    # signal / FFT
    n_samples: int = 2048
    duration: float = 1.0
    f_min: float = 1.0
    f_max: float = 960.0  # Nyquist = n_samples / (2*duration) = 1024; leave margin

    # Frequency parameterization:
    #   "global": legacy full-band sigmoid. Every (token, channel) frequency
    #     can move anywhere in [f_min, f_max].
    #   "anchored": channel c owns a fixed interval centered at
    #     f_min + (c + 0.5) * (f_max-f_min)/d_sine. Each token predicts a
    #     bounded softsign offset around that anchor. Frequency separation is
    #     then computed over tokens independently inside each channel.
    frequency_param_mode: str = "global"
    # Offset radius as a fraction of one channel interval. 0.48 leaves a 4%
    # guard gap between neighboring channel regions (about 6.4 Hz for d_sine=6).
    freq_anchor_radius_frac: float = 0.48

    # Encoder f-head positional bias spread. In anchored mode the same
    # length-aware bias is applied independently inside every channel region.
    f_bias_spread: float = 3.0

    # amplitude cap (softplus(raw_A).clamp(max=...)) to prevent runaway
    A_max: float = 10.0

    # frequency-separation auxiliary loss
    freq_sep_min_bins: float = 4.0  # main-lobe width for Hann ≈ 4 bins
    freq_sep_lambda: float = 0.5

    # training
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    max_steps: int = 50000
    grad_clip: float = 1.2
    log_every: int = 50
    val_every: int = 1000
    val_batches: int = 50
    ckpt_every: int = 1000
    ckpt_dir: str = "all-training/e2e-train/"
    # Freeze the encoder (and its tied token_emb output projection) during AE
    # training. Intended use: load a CLIP-trained checkpoint via --init-from
    # and train only the decoder to reconstruct from the frozen waveforms.
    freeze_encoder: bool = False

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
    # 3e-4 destabilized d_sine=6 after ~22k steps (pace-ice sweep); 1.5e-4
    # produced the best checkpoint and was still climbing at 38k.
    clip_lr: float = 1.5e-4
    clip_warmup_steps: int = 3000
    # 50k so the cosine schedule actually completes — the 38k best model
    # stopped mid-schedule (100k horizon) with the lr still high.
    clip_max_steps: int = 50000
    clip_logit_scale_init: float = 2.6593 # ln(1/0.07) — CLIP default
    clip_logit_scale_max: float = 4.6052  # ln(100) — clamp ceiling per CLIP
    clip_log_every: int = 100
    clip_val_every: int = 1000
    clip_val_batches: int = 50
    clip_ckpt_every: int = 2000
    clip_ckpt_dir: str = "all-training/freq-fix-d6-v2.5/"
    # Gradient caching (Gao et al. 2021). When set and < clip_batch_size, the
    # contrastive loss is computed across the full clip_batch_size of negatives
    # while only chunk_size examples are forwarded with grad at a time. Lets
    # you raise clip_batch_size (more negatives) without raising peak memory.
    # None disables (single forward pass, current behavior).
    # chunk == batch also disables it (direct mode) — the right setting on
    # large-memory boxes (DGX Spark 128 GB): one fewer forward pass per step.
    # Set back to 32 for memory-constrained machines, or raise clip_batch_size
    # above this to re-enable chunking automatically.
    clip_cache_chunk_size: int = 512
    # Execution-only length bucketing inside each already-sampled batch. This
    # does NOT change batch membership or InfoNCE negatives: it trims padding
    # for encoder/synthesis calls, then restores original row order. A width of
    # 16 retains large GEMMs while avoiding the ~6x padding waste of this data.
    clip_length_bucket_size: int = 16
    # Multi-source contrastive mix. Each entry is (hf_dataset_name, config_name);
    # use "" for datasets without a config. When non-empty, this overrides the
    # legacy single-source clip_dataset_name / clip_dataset_config. Empty tuple
    # = use the legacy single-source path (clip_dataset_name + clip_dataset_config).
    clip_dataset_specs: tuple = (
        ("sentence-transformers/all-nli", "pair"),
        ("sentence-transformers/quora-duplicates", "pair"),
        ("sentence-transformers/altlex", ""),
    )
    # Sentence-representation method. "spectral" is the project's spectral
    # autoencoder path: encoder → (A, f, φ) → synthesize → flatten/rfft. The
    # baselines pool the encoder's post-LN hidden states directly and skip
    # synthesis entirely; the sine-parameter head and waveform decoder are not
    # allocated in those modes. clip_per_channel_lambda / clip_recon_lambda /
    # freq_sep_lambda MUST be 0 when this is not "spectral".
    #   "spectral", "mean_pool", "cls", "max_pool"
    clip_encoder_mode: str = "spectral"
    # Embedding for the contrastive loss in spectral mode. "time": flatten the
    # synthesized waveform directly (default). "spectral": take |rfft(signal)|
    # per channel before flatten + L2-normalize — phase-invariant; dimensionality
    # drops to (N//2+1) * d_sine. Ignored when clip_encoder_mode != "spectral".
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

    # runtime. "auto" prefers CUDA, then MPS, then CPU. The freq-fix-d6 run is
    # intended for the DGX Spark, where BF16 activates Blackwell tensor cores;
    # non-CUDA devices safely fall back to FP32.
    device: str = "auto"
    precision: str = "bf16"  # "fp32" or "bf16" (CUDA only)
    cuda_tf32: bool = True
    fused_optimizer: bool = True
    # Keep the ARM CPU side ahead of the GPU and overlap page-locked transfers.
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    seed: int = 0

    def __post_init__(self):
        valid_modes = {"spectral", "mean_pool", "cls", "max_pool"}
        if self.clip_encoder_mode not in valid_modes:
            raise ValueError(
                f"clip_encoder_mode={self.clip_encoder_mode!r} not in {sorted(valid_modes)}"
            )
        valid_sine_modes = {"independent", "shared"}
        if self.sine_param_mode not in valid_sine_modes:
            raise ValueError(
                f"sine_param_mode={self.sine_param_mode!r} not in {sorted(valid_sine_modes)}"
            )
        valid_frequency_modes = {"global", "anchored"}
        if self.frequency_param_mode not in valid_frequency_modes:
            raise ValueError(
                f"frequency_param_mode={self.frequency_param_mode!r} not in "
                f"{sorted(valid_frequency_modes)}"
            )
        if not 0.0 < self.freq_anchor_radius_frac < 0.5:
            raise ValueError(
                "freq_anchor_radius_frac must be strictly between 0 and 0.5, "
                f"got {self.freq_anchor_radius_frac}"
            )
        if self.frequency_param_mode == "anchored" and self.sine_param_mode == "shared":
            raise ValueError(
                "frequency_param_mode='anchored' requires sine_param_mode='independent': "
                "fixed channel anchors imply one frequency per channel"
            )
        valid_precisions = {"fp32", "bf16"}
        if self.precision not in valid_precisions:
            raise ValueError(
                f"precision={self.precision!r} not in {sorted(valid_precisions)}"
            )


def config_from_snapshot(data: dict) -> Config:
    """Load a stored config without changing legacy checkpoint semantics.

    Snapshots written before frequency_param_mode existed used the global
    sigmoid parameterization. Config() defaults to the new anchored v3 recipe,
    so old snapshots must opt back into "global" explicitly when rehydrated.
    """
    values = dict(data)
    values.setdefault("frequency_param_mode", "global")
    return Config(**values)
