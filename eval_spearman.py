"""Evaluate a CLIP-trained encoder on the STS Spearman benchmark.

Usage:
    python eval_spearman.py <ckpt.pt>            # STS-B test only (default)
    python eval_spearman.py <ckpt.pt> --all      # Full STS12-16 + STS-B + SICK-R suite

Loads the encoder from a CLIP checkpoint, encodes each sentence pair, and
reports the Spearman + Pearson correlation between cosine similarity and the
gold human similarity score (multiplied by 100, the standard convention).
"""
import argparse

import torch
import torch.nn.functional as F

from config import Config
from model import SpectralAE
from train_clip import encode_to_embedding, pick_device


# (hf_name, split). All these datasets use sentence1/sentence2/score columns.
DEFAULT_DATASETS = [
    ("mteb/stsbenchmark-sts", "test"),
]
ALL_DATASETS = [
    ("mteb/sts12-sts", "test"),
    ("mteb/sts13-sts", "test"),
    ("mteb/sts14-sts", "test"),
    ("mteb/sts15-sts", "test"),
    ("mteb/sts16-sts", "test"),
    ("mteb/stsbenchmark-sts", "test"),
    ("mteb/sickr-sts", "test"),
]


def encode_sentences(model, tokenizer, sentences, cfg, device, batch_size):
    """Tokenize + encode a flat list of sentences. Returns CPU tensor (N, D)."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    embs = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        token_lists = []
        for s in chunk:
            ids = tokenizer(
                s or "",
                add_special_tokens=False,
                truncation=True,
                max_length=cfg.clip_max_len,
            )["input_ids"]
            if not ids:
                # STS rarely has empty strings, but guard anyway.
                ids = [pad_id]
            token_lists.append(ids)
        L = max(len(t) for t in token_lists)
        B = len(chunk)
        tokens = torch.full((B, L), pad_id, dtype=torch.long)
        mask = torch.ones((B, L), dtype=torch.bool)
        for j, t in enumerate(token_lists):
            tokens[j, : len(t)] = torch.tensor(t, dtype=torch.long)
            mask[j, : len(t)] = False
        tokens = tokens.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            emb, _ = encode_to_embedding(model, tokens, mask, cfg)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def evaluate_dataset(name, split, model, tokenizer, cfg, device, batch_size):
    from datasets import load_dataset
    from scipy.stats import pearsonr, spearmanr

    ds = load_dataset(name, split=split)
    cols = ds.column_names
    if "sentence1" not in cols or "sentence2" not in cols:
        raise RuntimeError(f"Unexpected columns in {name}: {cols}")
    if "score" in cols:
        gold = list(ds["score"])
    elif "label" in cols:
        gold = list(ds["label"])
    else:
        raise RuntimeError(f"No score/label column in {name}: {cols}")

    e1 = encode_sentences(model, tokenizer, list(ds["sentence1"]), cfg, device, batch_size)
    e2 = encode_sentences(model, tokenizer, list(ds["sentence2"]), cfg, device, batch_size)
    cos = F.cosine_similarity(e1, e2, dim=-1).numpy()
    rho = spearmanr(cos, gold).statistic
    r = pearsonr(cos, gold).statistic
    return len(gold), rho, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to CLIP-mode checkpoint .pt file")
    p.add_argument("--device", default="mps")
    p.add_argument("--all", action="store_true",
                   help="Run STS12-16 + STS-B + SICK-R (default: STS-B test only)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for sentence encoding (default: 64)")
    args = p.parse_args()

    from transformers import AutoTokenizer

    device = pick_device(args.device)
    print(f"[eval] device={device}  ckpt={args.ckpt}")
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    step = blob.get("step", -1)
    print(f"[eval] step={step}  d_sine={cfg.d_sine}  n_samples={cfg.n_samples}  "
          f"clip_max_len={cfg.clip_max_len}")

    datasets = ALL_DATASETS if args.all else DEFAULT_DATASETS
    print()
    print(f"  {'dataset':<32s} {'N':>6s}  {'Spearman':>8s}  {'Pearson':>8s}")
    print(f"  {'-'*32} {'-'*6}  {'-'*8}  {'-'*8}")
    results = []
    for name, split in datasets:
        n, rho, r = evaluate_dataset(name, split, model, tokenizer, cfg, device, args.batch_size)
        results.append((name, n, rho, r))
        print(f"  {name:<32s} {n:>6d}  {rho*100:>8.2f}  {r*100:>8.2f}")

    if len(results) > 1:
        avg_rho = sum(x[2] for x in results) / len(results)
        avg_r = sum(x[3] for x in results) / len(results)
        print(f"  {'-'*32} {'-'*6}  {'-'*8}  {'-'*8}")
        print(f"  {'average':<32s} {'':>6s}  {avg_rho*100:>8.2f}  {avg_r*100:>8.2f}")


if __name__ == "__main__":
    main()
