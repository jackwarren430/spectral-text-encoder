import os

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _tokenize_split(tokenizer, texts, batch_size=1000):
    ids = []
    for i in range(0, len(texts), batch_size):
        chunk = [t for t in texts[i : i + batch_size] if t]
        if not chunk:
            continue
        out = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for row in out:
            ids.extend(row)
    return ids


def build_token_chunks(cfg, split: str):
    """Tokenize a wikitext split and reshape into fixed-length chunks.

    Returns a LongTensor of shape (num_chunks, seq_len). Caches to disk because
    full wikitext-103 tokenization is slow.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f"{cfg.dataset_config}_{cfg.tokenizer_name}_{split}_L{cfg.seq_len}.pt"
    )
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=split)
    ids = _tokenize_split(tokenizer, ds["text"])
    n_chunks = len(ids) // cfg.seq_len
    ids = ids[: n_chunks * cfg.seq_len]
    chunks = torch.tensor(ids, dtype=torch.long).view(n_chunks, cfg.seq_len)
    torch.save(chunks, cache_path)
    return chunks


class ChunkDataset(Dataset):
    def __init__(self, chunks: torch.Tensor):
        self.chunks = chunks

    def __len__(self):
        return self.chunks.size(0)

    def __getitem__(self, idx):
        return self.chunks[idx]


def make_loaders(cfg):
    train_chunks = build_token_chunks(cfg, "train")
    val_chunks = build_token_chunks(cfg, "validation")
    train_loader = DataLoader(
        ChunkDataset(train_chunks),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    val_loader = DataLoader(
        ChunkDataset(val_chunks),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    return train_loader, val_loader
