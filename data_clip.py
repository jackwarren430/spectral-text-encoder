import os

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def build_nli_pairs(cfg):
    """Tokenize anchor/positive sentence pairs from the configured NLI dataset.

    Returns dict with:
        'a': list[list[int]]  — anchor token ids per example
        'p': list[list[int]]  — positive token ids per example
        'pad_id': int          — token id used for padding in the collate fn
    Both lists are variable-length, capped at cfg.clip_max_len.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = cfg.clip_dataset_name.replace("/", "_")
    cache_path = os.path.join(
        CACHE_DIR,
        f"{safe_name}_{cfg.clip_dataset_config}_{cfg.tokenizer_name}_max{cfg.clip_max_len}.pt",
    )
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # GPT-2 has no pad token; reuse eos. The pad positions will be masked
        # out by pad_mask so the actual id value never affects the result.
        pad_id = tokenizer.eos_token_id

    ds = load_dataset(cfg.clip_dataset_name, cfg.clip_dataset_config, split="train")
    cols = ds.column_names
    if "anchor" in cols and "positive" in cols:
        a_col, p_col = "anchor", "positive"
    elif "sentence1" in cols and "sentence2" in cols:
        a_col, p_col = "sentence1", "sentence2"
    else:
        raise RuntimeError(f"Unexpected NLI dataset columns: {cols}")

    a_ids = []
    p_ids = []
    for a, p in zip(ds[a_col], ds[p_col]):
        if not a or not p:
            continue
        ai = tokenizer(a, add_special_tokens=False, truncation=True, max_length=cfg.clip_max_len)["input_ids"]
        pi = tokenizer(p, add_special_tokens=False, truncation=True, max_length=cfg.clip_max_len)["input_ids"]
        if not ai or not pi:
            continue
        a_ids.append(ai)
        p_ids.append(pi)

    blob = {"a": a_ids, "p": p_ids, "pad_id": int(pad_id)}
    torch.save(blob, cache_path)
    return blob


class PairDataset(Dataset):
    def __init__(self, a, p):
        assert len(a) == len(p)
        self.a = a
        self.p = p

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return self.a[idx], self.p[idx]


class PairCollate:
    """Pads each side of a batch of (a, p) token-id pairs to its own L_max.

    Defined at module scope (not as a closure) so it can be pickled and shipped
    to DataLoader worker processes.
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def _pad(self, seqs):
        L = max(len(s) for s in seqs)
        B = len(seqs)
        tokens = torch.full((B, L), self.pad_id, dtype=torch.long)
        mask = torch.ones((B, L), dtype=torch.bool)  # True = padding
        for i, s in enumerate(seqs):
            n = len(s)
            tokens[i, :n] = torch.tensor(s, dtype=torch.long)
            mask[i, :n] = False
        return tokens, mask

    def __call__(self, batch):
        a_list, p_list = zip(*batch)
        ta, ma = self._pad(a_list)
        tp, mp = self._pad(p_list)
        return ta, ma, tp, mp


def make_clip_loaders(cfg):
    blob = build_nli_pairs(cfg)
    a, p, pad_id = blob["a"], blob["p"], blob["pad_id"]
    n = len(a)
    n_val = max(1, int(n * cfg.clip_val_frac))
    n_train = n - n_val
    train_ds = PairDataset(a[:n_train], p[:n_train])
    val_ds = PairDataset(a[n_train:], p[n_train:])

    collate = PairCollate(pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.clip_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.clip_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate,
        pin_memory=False,
    )
    return train_loader, val_loader
