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
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    decoder_hidden: int = 256
    decoder_layers: int = 2

    # signal / FFT
    n_samples: int = 1024
    duration: float = 1.0
    f_min: float = 1.0
    f_max: float = 480.0  # Nyquist = n_samples / (2*duration) = 512; leave margin

    # training
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
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
