"""Encode, reconstruct, sample, interpolate, and band-edit a spectral VAE."""

import argparse
import os

import torch

from config_vae import load_vae_config
from model_vae import SpectralVAE
from run_utils import find_latest_ckpt
from train_vae import _autocast, configure_runtime, pick_device


def tokenize_text(tokenizer, text, cfg, device):
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=cfg.vae_max_length - 1,
    )["input_ids"]
    ids = ids + [cfg.eos_token_id]
    tokens = torch.full(
        (1, cfg.vae_max_length), cfg.pad_token_id, dtype=torch.long, device=device
    )
    pad_mask = torch.ones(
        1, cfg.vae_max_length, dtype=torch.bool, device=device
    )
    tokens[0, : len(ids)] = torch.tensor(ids, device=device)
    pad_mask[0, : len(ids)] = False
    return tokens, pad_mask


def decode_rows(tokenizer, token_rows, eos_token_id):
    texts = []
    for row in token_rows.tolist():
        if eos_token_id in row:
            row = row[: row.index(eos_token_id)]
        texts.append(tokenizer.decode(row))
    return texts


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Spectral-VAE run folder")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text to encode and reconstruct")
    source.add_argument("--prior", type=int, help="Number of prior waveforms to sample")
    parser.add_argument(
        "--interpolate-text",
        help="Interpolate the canonical waveform toward this second text",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sample-posterior", action="store_true")
    parser.add_argument("--band", type=int, default=None)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-waveform", default=None)
    args = parser.parse_args()

    cfg = load_vae_config(args.run_dir)
    if args.device:
        cfg.device = args.device
    device = pick_device(cfg.device)
    configure_runtime(cfg, device)
    checkpoint = args.checkpoint or find_latest_ckpt(args.run_dir)
    if checkpoint is None:
        raise SystemExit(f"no step_*.pt checkpoint found in {args.run_dir}")
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SpectralVAE(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if args.prior is not None:
        if args.prior < 1:
            raise SystemExit("--prior must be positive")
        with _autocast(cfg, device):
            waveform = model.prior_waveform(args.prior, device=device)
        label = f"prior samples: {args.prior}"
    else:
        tokens, pad_mask = tokenize_text(tokenizer, args.text, cfg, device)
        with _autocast(cfg, device):
            waveform = model.encode_waveform(
                tokens, pad_mask, sample=args.sample_posterior
            )
        label = f"input: {args.text}"
        if args.interpolate_text:
            second_tokens, second_mask = tokenize_text(
                tokenizer, args.interpolate_text, cfg, device
            )
            with _autocast(cfg, device):
                second = model.encode_waveform(
                    second_tokens, second_mask, sample=False
                )
            waveform = model.interpolate_waveforms(waveform, second, args.alpha)
            label += f"\ninterpolate: alpha={args.alpha:g} -> {args.interpolate_text}"

    before_spectrum = torch.fft.rfft(waveform.squeeze(-1).float(), norm="ortho")
    before_energy = model.latent_layout.band_energies(before_spectrum).mean(0)
    if args.band is not None:
        waveform = model.edit_band_gain(waveform, args.band, args.gain)
        label += f"\nedit: band={args.band} gain={args.gain:g}"
    with _autocast(cfg, device):
        logits = model.decode_waveform(waveform)
    decoded = decode_rows(tokenizer, logits.argmax(dim=-1), cfg.eos_token_id)
    after_spectrum = torch.fft.rfft(waveform.squeeze(-1).float(), norm="ortho")
    after_energy = model.latent_layout.band_energies(after_spectrum).mean(0)

    print(f"checkpoint: {os.path.basename(checkpoint)}")
    print(label)
    for index, text in enumerate(decoded):
        print(f"decoded[{index}]: {text!r}")
    print("log-band-energy before:", [round(float(x), 4) for x in before_energy])
    print("log-band-energy after: ", [round(float(x), 4) for x in after_energy])
    if args.save_waveform:
        torch.save(waveform.detach().cpu(), args.save_waveform)
        print(f"saved waveform: {args.save_waveform}")


if __name__ == "__main__":
    main()
