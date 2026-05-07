import argparse
import os

import torch
from transformers import AutoTokenizer

from config import Config
from model import SpectralAE, synthesize


# Fixed 64-token sample (GPT-2 BPE) for consistent qualitative inspection
# across checkpoints. Tokenizes to 72 tokens; truncated to seq_len at runtime.
SAMPLE_TEXT = (
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic "
    "divisions. It extends from the Arctic Ocean in the north to the Southern "
    "Ocean in the south, and is bounded by the continents of Asia and "
    "Australia in the west and the Americas in the east, covering about 46 "
    "percent of Earth's water surface and roughly one third of its total "
    "surface area."
)


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_model(ckpt_path: str, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**blob["cfg"])
    model = SpectralAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, cfg, blob.get("step", -1)


def encode_sample(text: str, cfg: Config, device: str):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < cfg.seq_len:
        raise ValueError(
            f"Sample text only tokenizes to {len(ids)} tokens; need >= {cfg.seq_len}."
        )
    ids = ids[: cfg.seq_len]
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    return tok, tokens


def plot_waveform(signal: torch.Tensor, cfg: Config, out_path: str) -> None:
    """signal: (1, N, d_sine). Plots one subplot per channel sharing a time axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = signal[0].detach().cpu().numpy()  # (N, d_sine)
    N, d = sig.shape
    t = (torch.arange(N) * (cfg.duration / N)).numpy()

    fig, axes = plt.subplots(d, 1, figsize=(10, max(2, 1.2 * d)), sharex=True)
    if d == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(t, sig[:, j], linewidth=0.6)
        ax.set_ylabel(f"ch {j}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Combined sine waveform — {d} channels, {N} samples, duration {cfg.duration}s")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run(ckpt_path: str, text: str, device: str, plot_path: str | None):
    device = pick_device(device)
    print(f"[infer] device={device}  ckpt={ckpt_path}")
    model, cfg, step = load_model(ckpt_path, device)
    print(f"[infer] checkpoint step={step}  seq_len={cfg.seq_len}  vocab={cfg.vocab_size}")

    tok, tokens = encode_sample(text, cfg, device)

    with torch.no_grad():
        logits, targets, aux = model(tokens)
        # Re-run encoder + synthesize to grab the waveform for plotting. Cheap
        # for a single sample; avoids threading an extra return through the model.
        A, f, phi = model.encoder(tokens)
        signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    pred_ids = logits.argmax(-1)  # (1, L)

    correct = (pred_ids == targets).sum().item()
    total = targets.numel()
    print(f"[infer] reconstruction accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"[infer] freq-separation aux: {aux.item():.4f}")

    print("\n--- ORIGINAL ---")
    print(tok.decode(targets[0].tolist()))
    print("\n--- RECONSTRUCTED ---")
    print(tok.decode(pred_ids[0].tolist()))

    print("\n--- TOKEN-BY-TOKEN ---")
    print(f"{'idx':>3}  {'orig':<18} {'pred':<18}  hit")
    for i, (o, p) in enumerate(zip(targets[0].tolist(), pred_ids[0].tolist())):
        ot = tok.decode([o]).replace("\n", "\\n")
        pt = tok.decode([p]).replace("\n", "\\n")
        mark = "✓" if o == p else " "
        print(f"{i:>3}  {ot!r:<18} {pt!r:<18}  {mark}")

    if plot_path is None:
        base, _ = os.path.splitext(ckpt_path)
        plot_path = f"{base}_waveform.png"
    plot_waveform(signal, cfg, plot_path)
    print(f"\n[infer] saved waveform plot → {plot_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", help="path to checkpoint .pt file")
    p.add_argument("--text", default=SAMPLE_TEXT, help="override the sample text")
    p.add_argument("--device", default="mps")
    p.add_argument("--plot-path", default=None, help="output path for waveform plot (default: <ckpt>_waveform.png)")
    args = p.parse_args()
    run(args.ckpt, args.text, args.device, args.plot_path)


if __name__ == "__main__":
    main()
