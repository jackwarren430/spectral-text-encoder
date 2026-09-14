"""Compare multichannel and summed readouts of a spectral CLIP checkpoint.

The encoder is run once per validation batch. Each requested channel mode is
then evaluated on exactly the same in-batch retrieval candidate pools. For an
anchored model the script also measures cross-band dot-product interference
and how accurately ideal FFT band-pass filters recover the original channels
from their scalar sum.

Usage:
    python eval_channel_sum.py checkpoint.pt --device cuda
    python eval_channel_sum.py checkpoint.pt --batches 50 --output results.json
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from config import config_from_snapshot
from model import (
    SpectralAE,
    frequency_anchor_radius,
    frequency_anchors,
)
from train_clip import (
    _autocast,
    _move_batch,
    contrastive_loss,
    encode_to_signal,
    pick_device,
    signal_to_embedding,
)


def _new_mode_stats():
    return {"ce_sum": 0.0, "correct": 0, "count": 0}


def _new_sum_stats(d_sine):
    return {
        "matrix_matched_sq": 0.0,
        "matrix_cross_sq": 0.0,
        "positive_matched_sq": 0.0,
        "positive_cross_sq": 0.0,
        "self_cross_sq": 0.0,
        "self_multi_energy_sum": 0.0,
        "self_count": 0,
        "recover_cos_sum": 0.0,
        "recover_rel_l2_sum": 0.0,
        "recover_count": 0,
        "channel_energy": [0.0] * d_sine,
    }


def _accumulate_interference(stats, sig_a, sig_b):
    """Accumulate raw-dot cross-channel terms for one paired batch."""
    # matched[i,j] = sum_c <a[i,c], b[j,c]>
    matched = sig_a.new_zeros((sig_a.size(0), sig_b.size(0)))
    for c in range(sig_a.size(-1)):
        matched.add_(sig_a[:, :, c] @ sig_b[:, :, c].T)
    summed_dot = sig_a.sum(dim=-1) @ sig_b.sum(dim=-1).T
    cross = summed_dot - matched
    stats["matrix_matched_sq"] += matched.square().sum().item()
    stats["matrix_cross_sq"] += cross.square().sum().item()
    stats["positive_matched_sq"] += matched.diagonal().square().sum().item()
    stats["positive_cross_sq"] += cross.diagonal().square().sum().item()

    for signal in (sig_a, sig_b):
        multi_energy = signal.square().sum(dim=(1, 2))
        summed_energy = signal.sum(dim=-1).square().sum(dim=1)
        self_cross = summed_energy - multi_energy
        stats["self_cross_sq"] += self_cross.square().sum().item()
        stats["self_multi_energy_sum"] += multi_energy.sum().item()
        stats["self_count"] += signal.size(0)
        energy = signal.square().sum(dim=(0, 1)).tolist()
        for c, value in enumerate(energy):
            stats["channel_energy"][c] += value


def _accumulate_recoverability(stats, signal, cfg):
    """Recover pre-sum channels with fixed ideal masks for anchored bands."""
    if getattr(cfg, "frequency_param_mode", "global") != "anchored":
        return
    n_samples = signal.size(1)
    spec = torch.fft.rfft(signal.sum(dim=-1), dim=1)
    hz = torch.fft.rfftfreq(
        n_samples,
        d=cfg.duration / n_samples,
        device=signal.device,
    )
    anchors = frequency_anchors(cfg, signal.device, signal.dtype)
    radius = frequency_anchor_radius(cfg)
    recovered = []
    for anchor in anchors:
        mask = (hz >= anchor - radius) & (hz <= anchor + radius)
        band = torch.fft.irfft(spec * mask.unsqueeze(0), n=n_samples, dim=1)
        recovered.append(band)
    recovered = torch.stack(recovered, dim=-1)

    original = signal.transpose(1, 2).reshape(-1, n_samples)
    estimate = recovered.transpose(1, 2).reshape(-1, n_samples)
    cosine = F.cosine_similarity(original, estimate, dim=-1)
    rel_l2 = (estimate - original).norm(dim=-1) / original.norm(dim=-1).clamp(min=1e-12)
    stats["recover_cos_sum"] += cosine.sum().item()
    stats["recover_rel_l2_sum"] += rel_l2.sum().item()
    stats["recover_count"] += cosine.numel()


def _finalize_sum_stats(stats):
    def rms_ratio(num, den):
        return math.sqrt(num / max(den, 1e-30))

    self_count = max(1, stats["self_count"])
    mean_energy = stats["self_multi_energy_sum"] / self_count
    total_channel_energy = sum(stats["channel_energy"])
    return {
        "cross_term_rms_ratio_all_pairs": rms_ratio(
            stats["matrix_cross_sq"], stats["matrix_matched_sq"]
        ),
        "cross_term_rms_ratio_positives": rms_ratio(
            stats["positive_cross_sq"], stats["positive_matched_sq"]
        ),
        "self_cross_term_rms_ratio": (
            math.sqrt(stats["self_cross_sq"] / self_count) / max(mean_energy, 1e-30)
        ),
        "ideal_bandpass_recovery_cosine": (
            stats["recover_cos_sum"] / max(1, stats["recover_count"])
        ),
        "ideal_bandpass_recovery_rel_l2": (
            stats["recover_rel_l2_sum"] / max(1, stats["recover_count"])
        ),
        "channel_energy_share": [
            value / max(total_channel_energy, 1e-30)
            for value in stats["channel_energy"]
        ],
    }


@torch.no_grad()
def evaluate(ckpt, device, max_batches, modes, batch_size=None, num_workers=None):
    from data_clip import make_clip_loaders

    device = pick_device(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = config_from_snapshot(blob["cfg"])
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    if cfg.clip_encoder_mode != "spectral":
        raise RuntimeError("channel-sum evaluation requires a spectral checkpoint")

    if batch_size is not None:
        cfg.clip_batch_size = batch_size
    if num_workers is not None:
        cfg.num_workers = num_workers
        cfg.persistent_workers = num_workers > 0
    _, val_loader = make_clip_loaders(cfg, device=device)
    logit_scale = torch.as_tensor(
        blob.get("logit_scale", cfg.clip_logit_scale_init),
        device=device,
        dtype=torch.float32,
    )
    mode_stats = {mode: _new_mode_stats() for mode in modes}
    sum_stats = _new_sum_stats(cfg.d_sine)

    print(
        f"[channel-sum] device={device}  step={blob.get('step', -1)}  "
        f"batch={cfg.clip_batch_size}  batches={max_batches}  modes={','.join(modes)}"
    )
    used_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ta, ma, tp, mp = _move_batch(batch, device, cfg)
        with _autocast(cfg, device):
            sig_a, _ = encode_to_signal(model, ta, ma, cfg)
            sig_b, _ = encode_to_signal(model, tp, mp, cfg)
            for mode in modes:
                cfg.signal_channel_mode = mode
                emb_a = signal_to_embedding(sig_a, cfg)
                emb_b = signal_to_embedding(sig_b, cfg)
                ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)
                targets = torch.arange(emb_a.size(0), device=device)
                stats = mode_stats[mode]
                stats["ce_sum"] += ce.item() * emb_a.size(0)
                stats["correct"] += (logits.argmax(dim=-1) == targets).sum().item()
                stats["count"] += emb_a.size(0)

        # Diagnostics are intentionally FP32 and act on the pre-sum channels.
        _accumulate_interference(sum_stats, sig_a.float(), sig_b.float())
        _accumulate_recoverability(sum_stats, sig_a.float(), cfg)
        _accumulate_recoverability(sum_stats, sig_b.float(), cfg)
        used_batches += 1
        print(f"[channel-sum] evaluated batch {used_batches}/{max_batches}", end="\r")
    print()

    results = {
        "checkpoint": ckpt,
        "step": int(blob.get("step", -1)),
        "batch_size": cfg.clip_batch_size,
        "batches": used_batches,
        "candidate_pairs": next(iter(mode_stats.values()))["count"],
        "modes": {},
        "diagnostics": _finalize_sum_stats(sum_stats),
    }
    for mode, stats in mode_stats.items():
        count = max(1, stats["count"])
        results["modes"][mode] = {
            "ce": stats["ce_sum"] / count,
            "accuracy": stats["correct"] / count,
        }
    return results


def _print_results(results):
    print()
    print(f"  {'readout':<10s} {'CE':>9s} {'top-1':>9s}")
    print(f"  {'-' * 10} {'-' * 9} {'-' * 9}")
    for mode, values in results["modes"].items():
        print(f"  {mode:<10s} {values['ce']:>9.4f} {values['accuracy'] * 100:>8.2f}%")
    d = results["diagnostics"]
    print()
    print("  summed-channel diagnostics:")
    print(f"    cross-term RMS / matched RMS (all pairs): {d['cross_term_rms_ratio_all_pairs'] * 100:.2f}%")
    print(f"    cross-term RMS / matched RMS (positives): {d['cross_term_rms_ratio_positives'] * 100:.2f}%")
    print(f"    self cross-term RMS / channel energy:     {d['self_cross_term_rms_ratio'] * 100:.2f}%")
    print(f"    ideal-bandpass recovery cosine:           {d['ideal_bandpass_recovery_cosine']:.4f}")
    print(f"    ideal-bandpass recovery relative L2:       {d['ideal_bandpass_recovery_rel_l2']:.4f}")
    shares = ", ".join(f"{x * 100:.1f}%" for x in d["channel_energy_share"])
    print(f"    channel energy shares:                    [{shares}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="path to a spectral CLIP checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--modes", nargs="+", choices=["multi", "sum"], default=["multi", "sum"]
    )
    parser.add_argument("--output", default=None, help="optional JSON results path")
    args = parser.parse_args()

    results = evaluate(
        args.ckpt,
        args.device,
        args.batches,
        args.modes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _print_results(results)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[channel-sum] wrote {args.output}")


if __name__ == "__main__":
    main()
