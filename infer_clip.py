import argparse
import math
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import Config, config_from_snapshot
from model import SpectralAE, combine_signal_channels, synthesize
from train_clip import encode_to_embedding


DEFAULT_TEXT_A = "The cat sat on the mat."
DEFAULT_TEXT_B = "A feline rested on the rug."


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_clip_checkpoint(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = config_from_snapshot(blob["cfg"])
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    logit_scale = blob.get("logit_scale", None)
    if logit_scale is not None:
        logit_scale = torch.as_tensor(logit_scale, device=device, dtype=torch.float32)
    step = blob.get("step", -1)
    return model, cfg, logit_scale, step


def tokenize_one(tokenizer, text: str, max_len: int, device: str):
    """Returns (tokens, pad_mask) with tokens (1, L) and an all-False mask (1, L).

    The mask is technically unneeded for a single sentence, but threading it
    through ensures the encoder uses the per-row f-bias path it saw at training
    time (rather than the no-mask single-linspace path)."""
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    if len(ids) == 0:
        raise ValueError(f"Input tokenized to 0 tokens: {text!r}")
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    pad_mask = torch.zeros_like(tokens, dtype=torch.bool)
    return tokens, pad_mask


def plot_two_waveforms(sig_a: torch.Tensor, sig_b: torch.Tensor, cfg: Config, out_path: str,
                       label_a: str = "A", label_b: str = "B") -> None:
    """sig_a, sig_b: (1, N, d_sine). One subplot per channel; both overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = sig_a[0].detach().cpu().numpy()
    b = sig_b[0].detach().cpu().numpy()
    N, d = a.shape
    t = (torch.arange(N) * (cfg.duration / N)).numpy()

    fig, axes = plt.subplots(d, 1, figsize=(10, max(2, 1.4 * d)), sharex=True)
    if d == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(t, a[:, j], linewidth=0.6, label=label_a, color="tab:blue", alpha=0.8)
        ax.plot(t, b[:, j], linewidth=0.6, label=label_b, color="tab:orange", alpha=0.8)
        ax.set_ylabel(f"ch {j}")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Pair waveforms — {d} channels, {N} samples, duration {cfg.duration}s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_summed_waveforms(
    sig_a: torch.Tensor,
    sig_b: torch.Tensor,
    cfg: Config,
    out_path: str,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Plot the scalar summed-channel readout at full sample resolution.

    The two inputs are the raw ``(1, N, d_sine)`` synthesized tensors. This
    function always applies the summed readout itself, independent of a
    checkpoint or CLI channel-mode override, so the diagnostic is available
    for both legacy multichannel and summed-channel runs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sig_a.ndim != 3 or sig_b.ndim != 3 or sig_a.shape != sig_b.shape:
        raise ValueError(
            "Expected matching (B, N, d_sine) signals, got "
            f"{tuple(sig_a.shape)} and {tuple(sig_b.shape)}"
        )

    d_sine = sig_a.size(-1)
    summed_a = sig_a.sum(dim=-1) / math.sqrt(d_sine)
    summed_b = sig_b.sum(dim=-1) / math.sqrt(d_sine)
    a = summed_a[0].detach().float().cpu().numpy()
    b = summed_b[0].detach().float().cpu().numpy()
    n_samples = a.shape[0]
    t = (torch.arange(n_samples, dtype=torch.float64) * (cfg.duration / n_samples)).numpy()

    # A shared y scale makes amplitude/energy differences directly visible,
    # while separate panels keep high-frequency structure from being obscured
    # by an overlay. At 300 DPI this is a 5,400-pixel-wide image, comfortably
    # wider than the 2,048-sample waveform.
    peak = max(float(abs(a).max()), float(abs(b).max()))
    y_pad = max(peak * 0.05, 1e-6)
    y_lim = (-peak - y_pad, peak + y_pad)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, sharey=True)
    for ax, values, label, color in (
        (axes[0], a, label_a, "tab:blue"),
        (axes[1], b, label_b, "tab:orange"),
    ):
        ax.plot(t, values, linewidth=0.55, color=color, antialiased=True)
        ax.axhline(0.0, linewidth=0.45, color="black", alpha=0.45)
        ax.set_ylabel(f"{label} amplitude")
        ax.set_ylim(y_lim)
        ax.margins(x=0)
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        "Summed scalar waveforms "
        f"(sum / sqrt({d_sine})) — {n_samples} samples, duration {cfg.duration}s"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(
    ckpt_path: str,
    text_a: str,
    text_b: str,
    device: str,
    plot_path: str | None,
    channel_mode: str | None = None,
    sum_plot_path: str | None = None,
):
    device = pick_device(device)
    print(f"[infer_clip] device={device}  ckpt={ckpt_path}")
    model, cfg, logit_scale, step = load_clip_checkpoint(ckpt_path, device)
    if channel_mode is not None:
        cfg.signal_channel_mode = channel_mode
        print(f"[infer_clip] overriding signal_channel_mode → {channel_mode}")
    scale_str = f"{logit_scale.exp().item():.2f}" if logit_scale is not None else "n/a"
    print(
        f"[infer_clip] checkpoint step={step}  d_sine={cfg.d_sine}  "
        f"n_samples={cfg.n_samples}  channels={cfg.signal_channel_mode}  "
        f"logit_scale.exp()={scale_str}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    ta, ma = tokenize_one(tokenizer, text_a, cfg.clip_max_len, device)
    tb, mb = tokenize_one(tokenizer, text_b, cfg.clip_max_len, device)
    print(f"[infer_clip] L_a={ta.size(1)}  L_b={tb.size(1)}  (cap={cfg.clip_max_len})")

    spectral_mode = cfg.clip_encoder_mode == "spectral"
    with torch.no_grad():
        if spectral_mode:
            # Raw waveforms for plotting + magnitude inspection.
            A_a, f_a, phi_a = model.encoder(ta, pad_mask=ma)
            sig_a = synthesize(A_a, f_a, phi_a, cfg.n_samples, cfg.duration)
            A_b, f_b, phi_b = model.encoder(tb, pad_mask=mb)
            sig_b = synthesize(A_b, f_b, phi_b, cfg.n_samples, cfg.duration)
            visible_a = combine_signal_channels(sig_a, cfg)
            visible_b = combine_signal_channels(sig_b, cfg)
        else:
            sig_a = sig_b = visible_a = visible_b = None
        emb_a, _ = encode_to_embedding(model, ta, ma, cfg)
        emb_b, _ = encode_to_embedding(model, tb, mb, cfg)

    cos = (emb_a * emb_b).sum().item()
    print()
    print(f"  cosine(a, b)        = {cos:+.4f}   (range [-1, 1]; >0 means similar)")
    if logit_scale is not None:
        print(f"  scaled logit        = {cos * logit_scale.exp().item():+.4f}   (what the loss saw)")
    if spectral_mode:
        norm_a = visible_a.flatten(1).norm().item()
        norm_b = visible_b.flatten(1).norm().item()
        print(f"  ||signal_a||_2      = {norm_a:.3f}")
        print(f"  ||signal_b||_2      = {norm_b:.3f}")
    print()
    print(f"  text A: {text_a!r}")
    print(f"  text B: {text_b!r}")

    if not spectral_mode:
        print(f"\n[infer_clip] encoder mode={cfg.clip_encoder_mode} — no waveform to plot")
        return
    if plot_path is None:
        base, _ = os.path.splitext(ckpt_path)
        plot_path = f"{base}_pair_waveform.png"
    if sum_plot_path is None:
        plot_base, _ = os.path.splitext(plot_path)
        sum_plot_path = f"{plot_base}_summed_highres.png"
    plot_two_waveforms(visible_a, visible_b, cfg, plot_path, label_a="A", label_b="B")
    plot_summed_waveforms(sig_a, sig_b, cfg, sum_plot_path, label_a="A", label_b="B")
    print(f"\n[infer_clip] saved waveform plot → {plot_path}")
    print(f"[infer_clip] saved high-resolution summed plot → {sum_plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to CLIP-mode checkpoint .pt file")
    p.add_argument("--text-a", default=DEFAULT_TEXT_A)
    p.add_argument("--text-b", default=DEFAULT_TEXT_B)
    p.add_argument("--device", default="mps")
    p.add_argument("--plot-path", default=None,
                   help="output path for waveform plot (default: <ckpt>_pair_waveform.png)")
    p.add_argument(
        "--sum-plot-path",
        default=None,
        help=(
            "output path for the 300-DPI summed-channel plot "
            "(default: <plot-path>_summed_highres.png)"
        ),
    )
    p.add_argument("--channel-mode", choices=["multi", "sum"], default=None,
                   help="override the checkpoint's observable channel readout")
    args = p.parse_args()
    run(
        args.ckpt, args.text_a, args.text_b, args.device, args.plot_path,
        args.channel_mode, args.sum_plot_path,
    )


if __name__ == "__main__":
    main()
