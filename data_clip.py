import os

import torch
from torch.utils.data import DataLoader, Dataset

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# Columns that indicate (anchor, positive) pairs across the various
# sentence-transformers datasets. Order = priority.
_PAIR_COLUMN_CANDIDATES = [
    ("anchor", "positive"),
    ("sentence1", "sentence2"),
    ("question1", "question2"),
    ("title1", "title2"),          # Stack Exchange duplicate titles
    ("caption1", "caption2"),      # COCO / Flickr same-image captions
    ("premise", "hypothesis"),
    ("text", "simplified"),        # AltLex / sentence compression
]


def _detect_pair_columns(cols):
    for a_col, p_col in _PAIR_COLUMN_CANDIDATES:
        if a_col in cols and p_col in cols:
            return a_col, p_col
    return None, None


def _build_one_source(name, dataset_config, tokenizer, tokenizer_name, max_len):
    """Tokenize a single (anchor, positive) source. Cached per (name, config,
    tokenizer, max_len). Returns dict with 'a': list[list[int]], 'p': list[list[int]]."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_name = name.replace("/", "_")
    safe_conf = dataset_config or "default"
    cache_path = os.path.join(
        CACHE_DIR,
        f"{safe_name}_{safe_conf}_{tokenizer_name}_max{max_len}.pt",
    )
    if os.path.exists(cache_path):
        return torch.load(cache_path)

    from datasets import load_dataset

    if dataset_config:
        ds = load_dataset(name, dataset_config, split="train")
    else:
        ds = load_dataset(name, split="train")
    a_col, p_col = _detect_pair_columns(ds.column_names)
    if a_col is None:
        raise RuntimeError(
            f"Could not find a (anchor, positive) column pair in {name} "
            f"(config={dataset_config!r}). Columns: {ds.column_names}"
        )

    a_ids, p_ids = [], []
    for a, p in zip(ds[a_col], ds[p_col]):
        if not a or not p:
            continue
        ai = tokenizer(a, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        pi = tokenizer(p, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        if not ai or not pi:
            continue
        a_ids.append(ai)
        p_ids.append(pi)

    blob = {"a": a_ids, "p": p_ids}
    torch.save(blob, cache_path)
    return blob


def _resolve_specs(cfg):
    """Pick which (name, config) sources to load. Multi-source if
    cfg.clip_dataset_specs is non-empty; otherwise fall back to the legacy
    single-source fields."""
    specs = list(getattr(cfg, "clip_dataset_specs", None) or ())
    if specs:
        # Tolerate JSON-roundtripped specs being lists of lists.
        return [(s[0], s[1]) for s in specs]
    return [(cfg.clip_dataset_name, cfg.clip_dataset_config)]


def build_pair_sources(specs, tokenizer_name: str, max_len: int, log_prefix="data_clip"):
    """Build tokenized pair sources for any training objective.

    CLIP and VAE training use the same source data, but keep independent
    configuration and training entry points.  This small public helper owns the
    shared tokenization/cache behavior without making the VAE pretend to be a
    CLIP run.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # GPT-2 has no pad token; reuse eos. Pad positions are masked out by
        # pad_mask so the actual id value never affects the result.
        pad_id = tokenizer.eos_token_id

    sources = []
    for name, dconf in specs:
        blob = _build_one_source(name, dconf, tokenizer, tokenizer_name, max_len)
        sources.append({"name": name, "config": dconf, "a": blob["a"], "p": blob["p"]})
        print(f"[{log_prefix}] {name} ({dconf or 'default'}): {len(blob['a'])} pairs")
    eos_id = tokenizer.eos_token_id
    return sources, int(pad_id), (None if eos_id is None else int(eos_id))


def build_pairs(cfg):
    """Tokenize all configured sources, returning a list of per-source dicts and
    the shared pad-token id. Each source dict has 'name', 'a', 'p'."""
    sources, pad_id, _eos_id = build_pair_sources(
        _resolve_specs(cfg), cfg.tokenizer_name, cfg.clip_max_len
    )
    return sources, pad_id


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


def make_clip_loaders(cfg, device=None):
    sources, pad_id = build_pairs(cfg)

    # Hold out cfg.clip_val_frac from EACH source independently so the val set
    # stays representative across sources rather than collapsing onto whichever
    # source ends up at the tail of a concatenated list.
    train_a, train_p, val_a, val_p = [], [], [], []
    for src in sources:
        n = len(src["a"])
        n_val = max(1, int(n * cfg.clip_val_frac))
        n_train = n - n_val
        train_a.extend(src["a"][:n_train])
        train_p.extend(src["p"][:n_train])
        val_a.extend(src["a"][n_train:])
        val_p.extend(src["p"][n_train:])

    # Shuffle the val list with a fixed seed so val batches are mixed-source
    # (matching the cross-source composition of shuffled train batches). Without
    # this, val batches cluster by source and become artificially harder than
    # train, inflating the apparent train/val gap. Deterministic across runs.
    if len(sources) > 1:
        import random
        rng = random.Random(1337)
        idx = list(range(len(val_a)))
        rng.shuffle(idx)
        val_a = [val_a[i] for i in idx]
        val_p = [val_p[i] for i in idx]

    train_ds = PairDataset(train_a, train_p)
    val_ds = PairDataset(val_a, val_p)

    collate = PairCollate(pad_id)
    device_type = str(device or cfg.device).split(":", 1)[0]
    if device_type == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = bool(getattr(cfg, "pin_memory", False) and device_type == "cuda")
    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "collate_fn": collate,
        "pin_memory": pin_memory,
    }
    if cfg.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(getattr(cfg, "persistent_workers", False)),
            prefetch_factor=int(getattr(cfg, "prefetch_factor", 2)),
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.clip_batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.clip_batch_size,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    return train_loader, val_loader
