"""Test how additive the spectral encoder is.

For each labeled pair (a, b), compare:
    W_joint = wave("a b")                 — encode the concatenated sentence
    W_sum   = wave(a) + wave(b)           — sum the two individual waveforms

A perfectly additive encoder would have W_joint == W_sum. The gap reveals how
much the encoder relies on cross-sentence context (attention across the two
spans). Results are written to ./experiments/spectral_compositionality_<ts>/.

Usage:
    python compositionality_test.py <ckpt.pt>
    python compositionality_test.py <ckpt.pt> --pairs-file pairs.json --device cpu

Pairs JSON format:
    [
      {"category": "related",  "a": "...", "b": "..."},
      {"category": "unrelated", "a": "...", "b": "..."}
    ]

Only works for checkpoints with clip_encoder_mode="spectral" (baseline modes
have no waveform).
"""
import argparse
import csv
import json
import os
import time

import torch
from transformers import AutoTokenizer

from infer_clip import load_clip_checkpoint, pick_device
from model import synthesize


DEFAULT_PAIRS = [
    # Semantically related — paraphrases / near-paraphrases.
    {"category": "related", "a": "The cat sat on the mat.",
                             "b": "A feline rested on the rug."},
    {"category": "related", "a": "She bought a new car.",
                             "b": "He purchased a vehicle."},
    {"category": "related", "a": "The team won the championship.",
                             "b": "Our squad took home the trophy."},

    # Unrelated — different topics, no shared entities.
    {"category": "unrelated", "a": "Quantum physics is notoriously difficult.",
                               "b": "I really enjoy strawberry ice cream."},
    {"category": "unrelated", "a": "The election was hotly contested.",
                               "b": "The chef prepared a five-course dinner."},
    {"category": "unrelated", "a": "Birds migrate south in winter.",
                               "b": "Stock prices fluctuated all morning."},

    # Interacting — b's meaning depends on a (coreference, causal, narrative).
    {"category": "interacting", "a": "The man approached the dog.",
                                 "b": "It started barking loudly."},
    {"category": "interacting", "a": "The temperature dropped sharply.",
                                 "b": "Everyone reached for warm clothes."},
    {"category": "interacting", "a": "She entered the dim room.",
                                 "b": "She closed the door behind her."},
    {"category": "interacting", "a": "The window broke during the storm.",
                                 "b": "Rain poured onto the carpet."},
]


def tokenize_one(tokenizer, text, max_len, device):
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_len)["input_ids"]
    if not ids:
        raise ValueError(f"Text tokenized to 0 tokens: {text!r}")
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
    return tokens, pad_mask


def wave_of(model, tokens, pad_mask, cfg):
    """encoder → synthesize → (1, N, d_sine) waveform."""
    A, f, phi = model.encoder(tokens, pad_mask=pad_mask)
    return synthesize(A, f, phi, cfg.n_samples, cfg.duration)


def compare_waveforms(w_joint, w_sum):
    """All metrics over the full (N * d_sine) flattened signal.

    Returns a dict of scalar floats; per-channel L2 split out as a list."""
    j = w_joint[0]                       # (N, d_sine)
    s = w_sum[0]
    diff = j - s

    j_flat = j.reshape(-1)
    s_flat = s.reshape(-1)
    d_flat = diff.reshape(-1)

    l2_joint = j_flat.norm().item()
    l2_sum = s_flat.norm().item()
    l2_diff = d_flat.norm().item()
    rel_l2 = l2_diff / max(l2_joint, 1e-12)
    cos = torch.nn.functional.cosine_similarity(
        j_flat.unsqueeze(0), s_flat.unsqueeze(0)
    ).item()
    per_channel_l2 = diff.norm(dim=0).tolist()  # (d_sine,)
    return {
        "l2_joint": l2_joint,
        "l2_sum": l2_sum,
        "l2_diff": l2_diff,
        "rel_l2": rel_l2,
        "cosine": cos,
        "per_channel_l2": per_channel_l2,
    }


def plot_pair(w_joint, w_sum, cfg, out_path, title):
    """One subplot per d_sine channel; both waveforms overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = w_joint[0].detach().cpu().numpy()      # (N, d_sine)
    s = w_sum[0].detach().cpu().numpy()
    N, d = j.shape
    t = (torch.arange(N) * (cfg.duration / N)).numpy()

    fig, axes = plt.subplots(d, 1, figsize=(10, max(2, 1.4 * d)), sharex=True)
    if d == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        ax.plot(t, j[:, c], linewidth=0.6, color="tab:blue",
                label="wave(a+b)", alpha=0.85)
        ax.plot(t, s[:, c], linewidth=0.6, color="tab:orange",
                label="wave(a)+wave(b)", alpha=0.85)
        ax.set_ylabel(f"ch {c}")
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(ckpt_path, pairs, device, out_root):
    device = pick_device(device)
    print(f"[compositionality] device={device}  ckpt={ckpt_path}")
    model, cfg, _, step = load_clip_checkpoint(ckpt_path, device)
    if cfg.clip_encoder_mode != "spectral":
        raise RuntimeError(
            f"checkpoint mode={cfg.clip_encoder_mode!r}; this test only "
            f"applies to spectral checkpoints (baseline modes have no waveform)"
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_dir = os.path.join(out_root, f"spectral_compositionality_{ckpt_tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[compositionality] step={step}  d_sine={cfg.d_sine}  "
          f"n_samples={cfg.n_samples}  duration={cfg.duration}")
    print(f"[compositionality] writing → {run_dir}")

    results = []
    print()
    print(f"  {'#':>2}  {'category':<12s}  {'L|a|':>4s} {'L|b|':>4s} {'L|ab|':>5s}  "
          f"{'rel_L2':>7s} {'cosine':>7s}  pair")
    print(f"  {'-'*2}  {'-'*12}  {'-'*4} {'-'*4} {'-'*5}  {'-'*7} {'-'*7}  ----")
    for idx, p in enumerate(pairs):
        a, b = p["a"], p["b"]
        category = p.get("category", "uncategorized")
        joined = f"{a} {b}"

        ta, ma = tokenize_one(tokenizer, a, cfg.clip_max_len, device)
        tb, mb = tokenize_one(tokenizer, b, cfg.clip_max_len, device)
        tj, mj = tokenize_one(tokenizer, joined, cfg.clip_max_len, device)

        with torch.no_grad():
            w_a = wave_of(model, ta, ma, cfg)
            w_b = wave_of(model, tb, mb, cfg)
            w_joint = wave_of(model, tj, mj, cfg)
            w_sum = w_a + w_b

        metrics = compare_waveforms(w_joint, w_sum)
        print(f"  {idx:>2d}  {category:<12s}  "
              f"{ta.size(1):>4d} {tb.size(1):>4d} {tj.size(1):>5d}  "
              f"{metrics['rel_l2']:>7.4f} {metrics['cosine']:>+7.4f}  "
              f"{a!r} || {b!r}")

        plot_path = os.path.join(run_dir, f"pair_{idx:02d}_{category}.png")
        title = (f"[{category}] a: {a}\nb: {b}\n"
                 f"rel_L2={metrics['rel_l2']:.4f}  cosine={metrics['cosine']:+.4f}")
        plot_pair(w_joint, w_sum, cfg, plot_path, title)

        results.append({
            "idx": idx,
            "category": category,
            "a": a,
            "b": b,
            "joined": joined,
            "L_a": ta.size(1),
            "L_b": tb.size(1),
            "L_joined": tj.size(1),
            **{k: v for k, v in metrics.items() if k != "per_channel_l2"},
            "per_channel_l2": metrics["per_channel_l2"],
            "plot": os.path.basename(plot_path),
        })

    # Category summary.
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    print()
    print("  category-averaged divergence (lower = more additive):")
    print(f"    {'category':<12s}  {'n':>3s}  {'rel_L2':>7s}  {'cosine':>7s}")
    summary = []
    for cat, rs in by_cat.items():
        n = len(rs)
        avg_rel = sum(r["rel_l2"] for r in rs) / n
        avg_cos = sum(r["cosine"] for r in rs) / n
        print(f"    {cat:<12s}  {n:>3d}  {avg_rel:>7.4f}  {avg_cos:>+7.4f}")
        summary.append({"category": cat, "n": n, "avg_rel_l2": avg_rel, "avg_cosine": avg_cos})

    out_json = os.path.join(run_dir, "results.json")
    with open(out_json, "w") as fh:
        json.dump({
            "ckpt": ckpt_path,
            "step": step,
            "cfg": cfg.__dict__,
            "pairs": results,
            "category_summary": summary,
        }, fh, indent=2)
    print(f"\n[compositionality] wrote {out_json}")

    out_csv = os.path.join(run_dir, "results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "category", "L_a", "L_b", "L_joined",
                    "l2_joint", "l2_sum", "l2_diff", "rel_l2", "cosine", "a", "b"])
        for r in results:
            w.writerow([r["idx"], r["category"], r["L_a"], r["L_b"], r["L_joined"],
                        f"{r['l2_joint']:.6f}", f"{r['l2_sum']:.6f}",
                        f"{r['l2_diff']:.6f}", f"{r['rel_l2']:.6f}",
                        f"{r['cosine']:.6f}", r["a"], r["b"]])
    print(f"[compositionality] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to spectral CLIP checkpoint .pt file")
    p.add_argument("--pairs-file", default=None,
                   help="optional JSON file with [{category, a, b}, ...]; "
                        "uses a built-in default set when omitted")
    p.add_argument("--device", default="mps")
    p.add_argument("--out-root", default="./experiments",
                   help="parent dir for per-run output folder (default: ./experiments)")
    args = p.parse_args()

    if args.pairs_file:
        with open(args.pairs_file) as fh:
            pairs = json.load(fh)
        for r in pairs:
            if "a" not in r or "b" not in r:
                raise ValueError(f"pairs entry missing 'a'/'b': {r}")
    else:
        pairs = DEFAULT_PAIRS

    run(args.ckpt, pairs, args.device, args.out_root)


if __name__ == "__main__":
    main()
