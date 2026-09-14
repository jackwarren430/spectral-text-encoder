"""Shared, cached semantic-textual-similarity evaluation helpers."""

from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn.functional as F


@dataclass
class TokenizedSTSPairs:
    name: str
    split: str
    sentence1: list[list[int]]
    sentence2: list[list[int]]
    scores: list[float]
    pad_id: int

    def __post_init__(self):
        if not (len(self.sentence1) == len(self.sentence2) == len(self.scores)):
            raise ValueError("STS sentence and score arrays must have equal lengths")

    def __len__(self):
        return len(self.scores)


def load_tokenized_sts(name, split, tokenizer, max_len):
    """Load and tokenize an STS split once for repeated evaluations."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    required = {"sentence1", "sentence2"}
    if not required.issubset(ds.column_names):
        raise RuntimeError(f"Unexpected columns in {name}: {ds.column_names}")
    score_col = "score" if "score" in ds.column_names else "label"
    if score_col not in ds.column_names:
        raise RuntimeError(f"No score/label column in {name}: {ds.column_names}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    def tokenize(values):
        encoded = tokenizer(
            list(values),
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        # Preserve one real position for the unlikely empty sentence.
        return [ids if ids else [pad_id] for ids in encoded]

    return TokenizedSTSPairs(
        name=name,
        split=split,
        sentence1=tokenize(ds["sentence1"]),
        sentence2=tokenize(ds["sentence2"]),
        scores=[float(x) for x in ds[score_col]],
        pad_id=int(pad_id),
    )


def _encode_token_lists(
    model,
    token_lists,
    pad_id,
    cfg,
    device,
    batch_size,
    encode_fn,
    autocast_context=None,
):
    embeddings = []
    for start in range(0, len(token_lists), batch_size):
        chunk = token_lists[start : start + batch_size]
        width = max(len(ids) for ids in chunk)
        tokens = torch.full(
            (len(chunk), width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        pad_mask = torch.ones(
            (len(chunk), width),
            dtype=torch.bool,
            device=device,
        )
        for row, ids in enumerate(chunk):
            tokens[row, : len(ids)] = torch.as_tensor(ids, device=device)
            pad_mask[row, : len(ids)] = False
        context = autocast_context() if autocast_context is not None else nullcontext()
        with context:
            embedding, _ = encode_fn(model, tokens, pad_mask, cfg)
        embeddings.append(embedding.float().cpu())
    return torch.cat(embeddings, dim=0)


@torch.inference_mode()
def evaluate_tokenized_sts(
    pairs,
    model,
    cfg,
    device,
    batch_size,
    encode_fn,
    autocast_context=None,
):
    """Return Pearson/Spearman correlations while preserving model mode."""
    from scipy.stats import pearsonr, spearmanr

    was_training = model.training
    model.eval()
    try:
        left = _encode_token_lists(
            model,
            pairs.sentence1,
            pairs.pad_id,
            cfg,
            device,
            batch_size,
            encode_fn,
            autocast_context,
        )
        right = _encode_token_lists(
            model,
            pairs.sentence2,
            pairs.pad_id,
            cfg,
            device,
            batch_size,
            encode_fn,
            autocast_context,
        )
        cosine = F.cosine_similarity(left, right, dim=-1).numpy()
        spearman = float(spearmanr(cosine, pairs.scores).statistic)
        pearson = float(pearsonr(cosine, pairs.scores).statistic)
    finally:
        model.train(was_training)
    return {
        "count": len(pairs),
        "spearman": spearman,
        "pearson": pearson,
    }
