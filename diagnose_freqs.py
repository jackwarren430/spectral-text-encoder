"""Frequency-collapse diagnostic for a CLIP-mode checkpoint.

Answers three questions about the encoder's predicted frequencies on real
validation data:

1. Is the frequency transform saturated? For legacy global mode, invert the
   full-band sigmoid. For anchored mode, invert each channel's bounded
   softsign offset and report occupancy near its local region boundary.
2. Does f vary with the input at all? Per-slot std of raw_f (bias removed)
   across sentences. Near-zero means the model has abandoned frequency as an
   information channel and encodes only in A and phi.
3. How badly did pad slots pollute the old (unmasked) separation loss?
   Prints the aux value with and without the pad mask on the same batches.

Usage:
    conda run -n dl python diagnose_freqs.py all-training/.../step_38000.pt
"""
import argparse
import os

import torch

from config import Config, config_from_snapshot
from data_clip import make_clip_loaders
from model import (
    SpectralAE,
    f_init_bias,
    freq_separation_loss,
    freqs_for_separation,
    frequency_anchor_radius,
    frequency_anchors,
)

EPS = 1e-7           # inverse-transform clamp
SAT_THRESHOLD = 4.0  # local/global transform-input saturation threshold
EDGE_HZ = 20.0       # "cluster at the edge" = within this many Hz of f_min/f_max


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_checkpoint(ckpt_path: str, device: str):
    """Like infer_clip.load_clip_checkpoint, but infers sine_param_mode from
    the stored head shape. Old checkpoints predate the field, and rehydrating
    them with today's default ("shared") would fail to load an
    independent-mode state dict."""
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(blob["cfg"])
    head_out = blob["model"]["encoder.head.5.weight"].shape[0]
    d_sine = cfg_dict["d_sine"]
    if head_out == 3 * d_sine:
        inferred = "independent"
    elif head_out == d_sine + 2:
        inferred = "shared"
    else:
        raise RuntimeError(f"head_out={head_out} matches neither mode for d_sine={d_sine}")
    stored = cfg_dict.get("sine_param_mode")
    if stored is not None and stored != inferred:
        raise RuntimeError(f"stored sine_param_mode={stored!r} but head shape says {inferred!r}")
    cfg_dict["sine_param_mode"] = inferred
    cfg = config_from_snapshot(cfg_dict)
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, blob.get("step", -1)


@torch.no_grad()
def collect(model, cfg, loader, n_batches, device):
    """Run encoder on val batches (both pair sides). Returns dict of flat CPU
    tensors: pre-transform / f / raw_f at real slots, f at pad slots, pinned
    flags, per-unit raw_f groups for input-variation stats, and the aux loss
    computed the old (unmasked) and new (masked) way."""
    shared = cfg.sine_param_mode == "shared"
    anchored = getattr(cfg, "frequency_param_mode", "global") == "anchored"
    min_sep = cfg.freq_sep_min_bins / cfg.duration
    span = cfg.f_max - cfg.f_min

    pre_real, f_real, f_pad, pinned_real = [], [], [], []
    f_by_channel = [[] for _ in range(cfg.d_sine if not shared else 1)]
    raw_by_unit = {}  # (pos, ch) -> list of raw_f tensors across rows
    aux_old_sum, aux_new_sum, n_sides = 0.0, 0.0, 0

    done = 0
    for batch in loader:
        if done >= n_batches:
            break
        done += 1
        ta, ma, tp, mp = [x.to(device) for x in batch]
        for tokens, mask in [(ta, ma), (tp, mp)]:
            _, f, _ = model.encoder(tokens, pad_mask=mask)
            B, L, D = f.shape
            lengths = (~mask).sum(dim=-1)
            bias = f_init_bias(
                L, 1 if (shared or anchored) else D, cfg.f_bias_spread, device, f.dtype,
                real_lengths=lengths,
            )
            if shared:
                f_unit = f[..., :1]  # (B, L, 1) — one frequency per slot
            else:
                f_unit = f  # (B, L, D)
            if anchored:
                anchors = frequency_anchors(cfg, device, f.dtype).view(1, 1, D)
                p = (f_unit - anchors) / frequency_anchor_radius(cfg)
                pinned = p.abs() > 1 - EPS
                pre = p / (1.0 - p.abs()).clamp(min=EPS)  # inverse softsign
            else:
                p = (f_unit - cfg.f_min) / span
                pinned = (p < EPS) | (p > 1 - EPS)
                pre = torch.logit(p, eps=EPS)
            raw = pre - bias

            real = (~mask).unsqueeze(-1).expand_as(f_unit)
            pre_real.append(pre[real].cpu())
            f_real.append(f_unit[real].cpu())
            f_pad.append(f_unit[~real].cpu())
            pinned_real.append(pinned[real].cpu())
            for ch in range(f_unit.size(-1)):
                f_by_channel[ch].append(f_unit[:, :, ch][~mask].cpu())

            for pos in range(L):
                rows = ~mask[:, pos]
                if rows.sum() < 2:
                    continue
                vals = raw[rows, pos, :].cpu()  # (n_rows, units_at_pos)
                for ch in range(vals.shape[1]):
                    raw_by_unit.setdefault((pos, ch), []).append(vals[:, ch])

            fk_old, _ = freqs_for_separation(f, cfg)
            fk_new, valid = freqs_for_separation(f, cfg, mask)
            aux_old_sum += freq_separation_loss(fk_old, min_sep).item()
            aux_new_sum += freq_separation_loss(fk_new, min_sep, valid).item()
            n_sides += 1

    return {
        "pre": torch.cat(pre_real),
        "f": torch.cat(f_real),
        "f_pad": torch.cat(f_pad),
        "pinned": torch.cat(pinned_real),
        "f_by_channel": [torch.cat(parts) for parts in f_by_channel],
        "raw_by_unit": {k: torch.cat(v) for k, v in raw_by_unit.items()},
        "aux_old": aux_old_sum / max(1, n_sides),
        "aux_new": aux_new_sum / max(1, n_sides),
    }


def report(d, cfg):
    pre, f, f_pad, pinned = d["pre"], d["f"], d["f_pad"], d["pinned"]
    n = pre.numel()
    frac = lambda m: 100.0 * m.sum().item() / n

    anchored = getattr(cfg, "frequency_param_mode", "global") == "anchored"
    transform = "local softsign" if anchored else "global sigmoid"
    print(f"\n=== {transform} saturation (n={n} real waves, {f_pad.numel()} pad waves) ===")
    print(f"  |transform input| > {SAT_THRESHOLD:.0f}: {frac(pre.abs() > SAT_THRESHOLD):5.1f}%")
    print(f"  |transform input| > 6: {frac(pre.abs() > 6):5.1f}%")
    print(f"  numerically pinned  : {frac(pinned):5.1f}%")
    if anchored:
        ratio = pre.abs() / (1.0 + pre.abs())
        print(f"  > 90% local radius : {frac(ratio > 0.9):5.1f}%")

    lo, hi = cfg.f_min + EDGE_HZ, cfg.f_max - EDGE_HZ
    print(f"\n=== frequency distribution (Hz) ===")
    q = torch.quantile(f, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    print(f"  quantiles           : " + "  ".join(f"{v:.1f}" for v in q))
    print(f"  low edge  [{cfg.f_min:.0f}, {lo:.0f}]  : {frac(f < lo):5.1f}%")
    print(f"  middle    ({lo:.0f}, {hi:.0f}) : {frac((f >= lo) & (f <= hi)):5.1f}%")
    print(f"  high edge [{hi:.0f}, {cfg.f_max:.0f}]: {frac(f > hi):5.1f}%")
    if anchored:
        print("\n=== per-channel anchored regions (observed min / max Hz) ===")
        anchors = frequency_anchors(cfg, torch.device("cpu"), torch.float32)
        radius = frequency_anchor_radius(cfg)
        for ch, vals in enumerate(d["f_by_channel"]):
            print(
                f"  ch {ch}: allowed [{anchors[ch]-radius:7.2f}, {anchors[ch]+radius:7.2f}]"
                f"  observed [{vals.min():7.2f}, {vals.max():7.2f}]"
            )

    stds, means = [], []
    for vals in d["raw_by_unit"].values():
        if vals.numel() >= 10:
            stds.append(vals.std().item())
            means.append(vals.mean().item())
    stds_t, means_t = torch.tensor(stds), torch.tensor(means)
    print(f"\n=== input-dependence of raw_f (f_bias removed, {len(stds)} (slot,ch) units) ===")
    print(f"  mean per-unit std across inputs : {stds_t.mean():.4f}   (≈0 → f carries no per-input info)")
    print(f"  std of per-unit means           : {means_t.std():.4f}   (positional variation, for scale)")

    print(f"\n=== separation aux on these batches (min_sep={cfg.freq_sep_min_bins / cfg.duration:.1f} Hz) ===")
    print(f"  pads included      : {d['aux_old']:.4f}")
    print(f"  pads masked        : {d['aux_new']:.4f}")


def plot(d, cfg, step, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre, f, f_pad = d["pre"].numpy(), d["f"].numpy(), d["f_pad"].numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.hist(pre, bins=90, range=(-17, 17), color="tab:blue")
    for x0, x1 in [(-17, -SAT_THRESHOLD), (SAT_THRESHOLD, 17)]:
        ax1.axvspan(x0, x1, color="tab:red", alpha=0.08)
    for x in (-SAT_THRESHOLD, SAT_THRESHOLD):
        ax1.axvline(x, color="tab:red", linestyle="--", linewidth=1)
    sat = 100.0 * (abs(pre) > SAT_THRESHOLD).mean()
    transform = "softsign" if getattr(cfg, "frequency_param_mode", "global") == "anchored" else "sigmoid"
    ax1.set_title(f"pre-{transform} (raw_f + f_bias), real slots — {sat:.0f}% saturated")
    ax1.set_xlabel(f"pre-{transform} value")
    ax1.set_ylabel("count")
    ax1.grid(True, alpha=0.3)

    bins = 96
    ax2.hist(f, bins=bins, range=(0, cfg.f_max + 10), color="tab:blue", label="real slots")
    if f_pad.size:
        ax2.hist(f_pad, bins=bins, range=(0, cfg.f_max + 10), color="tab:orange",
                 alpha=0.6, label="pad slots (excluded by fix)")
    ax2.set_yscale("log")
    ax2.set_title("predicted frequencies (Hz)")
    ax2.set_xlabel("f (Hz)")
    ax2.set_ylabel("count (log)")
    ax2.legend(loc="upper center", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"frequency diagnostic — step {step}, {cfg.sine_param_mode}/"
        f"{getattr(cfg, 'frequency_param_mode', 'global')}, d_sine={cfg.d_sine}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\n[diagnose_freqs] saved plot → {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to CLIP-mode checkpoint .pt file")
    p.add_argument("--device", default="mps")
    p.add_argument("--batches", type=int, default=4, help="val batches (both pair sides used)")
    p.add_argument("--plot-path", default=None, help="default: <ckpt>_freq_diag.png")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"[diagnose_freqs] device={device}  ckpt={args.ckpt}")
    model, cfg, step = load_checkpoint(args.ckpt, device)
    print(f"[diagnose_freqs] step={step}  mode={cfg.sine_param_mode}/"
          f"{getattr(cfg, 'frequency_param_mode', 'global')}  d_sine={cfg.d_sine}  "
          f"f∈[{cfg.f_min}, {cfg.f_max}]  bias spread=±{cfg.f_bias_spread}")

    _, val_loader = make_clip_loaders(cfg)
    d = collect(model, cfg, val_loader, args.batches, device)
    report(d, cfg)

    out = args.plot_path or f"{os.path.splitext(args.ckpt)[0]}_freq_diag.png"
    plot(d, cfg, step, out)


if __name__ == "__main__":
    main()
