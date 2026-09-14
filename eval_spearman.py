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

from config import Config, config_from_snapshot
from model import SpectralAE
from sts_eval import evaluate_tokenized_sts, load_tokenized_sts
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


def evaluate_dataset(name, split, model, tokenizer, cfg, device, batch_size):
    pairs = load_tokenized_sts(name, split, tokenizer, cfg.clip_max_len)
    result = evaluate_tokenized_sts(
        pairs,
        model,
        cfg,
        device,
        batch_size,
        encode_to_embedding,
    )
    return result["count"], result["spearman"], result["pearson"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to CLIP-mode checkpoint .pt file")
    p.add_argument("--device", default="mps")
    p.add_argument("--all", action="store_true",
                   help="Run STS12-16 + STS-B + SICK-R (default: STS-B test only)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for sentence encoding (default: 64)")
    p.add_argument("--embedding-type", choices=["time", "spectral"], default=None,
                   help="Override cfg.clip_embedding_type for this eval run")
    p.add_argument("--channel-mode", choices=["multi", "sum"], default=None,
                   help="Override the checkpoint's observable channel readout")
    args = p.parse_args()

    from transformers import AutoTokenizer

    device = pick_device(args.device)
    print(f"[eval] device={device}  ckpt={args.ckpt}")
    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = config_from_snapshot(blob["cfg"])
    if "sine_param_mode" not in blob["cfg"]:
        # Checkpoints predating sine_param_mode used independent triples.
        cfg.sine_param_mode = "independent"
    # Construct with the checkpoint's architectural config before applying
    # readout-only overrides. This keeps an old multichannel decoder loadable
    # during a post-hoc summed evaluation (the decoder is not used here).
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    if args.embedding_type is not None:
        cfg.clip_embedding_type = args.embedding_type
        print(f"[eval] overriding clip_embedding_type → {cfg.clip_embedding_type}")
    if args.channel_mode is not None:
        cfg.signal_channel_mode = args.channel_mode
        print(f"[eval] overriding signal_channel_mode → {cfg.signal_channel_mode}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    step = blob.get("step", -1)
    print(f"[eval] step={step}  enc={cfg.clip_encoder_mode}  d_sine={cfg.d_sine}  "
          f"n_samples={cfg.n_samples}  channels={cfg.signal_channel_mode}  "
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
