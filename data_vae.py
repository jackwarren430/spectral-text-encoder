"""Sentence data for the standalone spectral VAE pipeline."""

import random

import torch
from torch.utils.data import DataLoader, Dataset

from data_clip import build_pair_sources


class SentenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class VAECollate:
    """Append EOS and always issue max-length tensors.

    Fixed output width is intentional: the decoder must infer EOS from the
    waveform rather than receiving a batch-dependent target length.
    """

    def __init__(self, max_length: int, pad_id: int, eos_id: int):
        self.max_length = max_length
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __call__(self, sequences):
        batch = torch.full(
            (len(sequences), self.max_length), self.pad_id, dtype=torch.long
        )
        pad_mask = torch.ones(len(sequences), self.max_length, dtype=torch.bool)
        for row, sequence in enumerate(sequences):
            ids = list(sequence[: self.max_length - 1]) + [self.eos_id]
            length = len(ids)
            batch[row, :length] = torch.tensor(ids, dtype=torch.long)
            pad_mask[row, :length] = False
        return batch, pad_mask


def make_vae_loaders(cfg, device=None):
    # Reuse the existing CLIP cache at this width; VAECollate truncates to
    # max_length - 1 before appending EOS, so the public sequence still has the
    # required explicit terminator without creating a second token cache.
    specs = list(cfg.vae_dataset_specs)
    sources, pad_id, eos_id = build_pair_sources(
        specs, cfg.tokenizer_name, cfg.vae_max_length, log_prefix="data_vae"
    )
    if eos_id != cfg.eos_token_id:
        raise ValueError(
            f"tokenizer EOS id {eos_id} does not match config {cfg.eos_token_id}"
        )
    if pad_id != cfg.pad_token_id:
        raise ValueError(
            f"tokenizer pad id {pad_id} does not match config {cfg.pad_token_id}"
        )

    train_sequences, val_sequences = [], []
    for source in sources:
        n_pairs = len(source["a"])
        n_val = max(1, int(n_pairs * cfg.vae_val_frac))
        split = n_pairs - n_val
        # Split pairs before flattening their sides, preventing pair partners
        # from being divided across train and validation.
        train_sequences.extend(source["a"][:split])
        train_sequences.extend(source["p"][:split])
        val_sequences.extend(source["a"][split:])
        val_sequences.extend(source["p"][split:])

    rng = random.Random(1337)
    rng.shuffle(val_sequences)
    collate = VAECollate(cfg.vae_max_length, pad_id, eos_id)
    device_type = str(device or cfg.device).split(":", 1)[0]
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    loader_args = {
        "num_workers": cfg.num_workers,
        "collate_fn": collate,
        "pin_memory": bool(cfg.pin_memory and device_type == "cuda"),
    }
    if cfg.num_workers > 0:
        loader_args.update(
            persistent_workers=bool(cfg.persistent_workers),
            prefetch_factor=int(cfg.prefetch_factor),
        )
    train_loader = DataLoader(
        SentenceDataset(train_sequences),
        batch_size=cfg.vae_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_args,
    )
    val_loader = DataLoader(
        SentenceDataset(val_sequences),
        batch_size=cfg.vae_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )
    return train_loader, val_loader
