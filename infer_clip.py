import argparse
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import Config
from model import SpectralAE, synthesize
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
    cfg = Config(**blob["cfg"])
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


def run(ckpt_path: str, text_a: str, text_b: str, device: str, plot_path: str | None):
    device = pick_device(device)
    print(f"[infer_clip] device={device}  ckpt={ckpt_path}")
    model, cfg, logit_scale, step = load_clip_checkpoint(ckpt_path, device)
    scale_str = f"{logit_scale.exp().item():.2f}" if logit_scale is not None else "n/a"
    print(
        f"[infer_clip] checkpoint step={step}  d_sine={cfg.d_sine}  "
        f"n_samples={cfg.n_samples}  logit_scale.exp()={scale_str}"
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
        else:
            sig_a = sig_b = None
        emb_a, _ = encode_to_embedding(model, ta, ma, cfg)
        emb_b, _ = encode_to_embedding(model, tb, mb, cfg)

    cos = (emb_a * emb_b).sum().item()
    print()
    print(f"  cosine(a, b)        = {cos:+.4f}   (range [-1, 1]; >0 means similar)")
    if logit_scale is not None:
        print(f"  scaled logit        = {cos * logit_scale.exp().item():+.4f}   (what the loss saw)")
    if spectral_mode:
        norm_a = sig_a.flatten(1).norm().item()
        norm_b = sig_b.flatten(1).norm().item()
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
    plot_two_waveforms(sig_a, sig_b, cfg, plot_path, label_a="A", label_b="B")
    print(f"\n[infer_clip] saved waveform plot → {plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to CLIP-mode checkpoint .pt file")
    p.add_argument("--text-a", default=DEFAULT_TEXT_A)
    p.add_argument("--text-b", default=DEFAULT_TEXT_B)
    p.add_argument("--device", default="mps")
    p.add_argument("--plot-path", default=None,
                   help="output path for waveform plot (default: <ckpt>_pair_waveform.png)")
    args = p.parse_args()
    run(args.ckpt, args.text_a, args.text_b, args.device, args.plot_path)


if __name__ == "__main__":
    main()
