import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from model import SpectralAE, synthesize


def lr_lambda(step, cfg: Config):
    if step < cfg.clip_warmup_steps:
        return step / max(1, cfg.clip_warmup_steps)
    progress = (step - cfg.clip_warmup_steps) / max(1, cfg.clip_max_steps - cfg.clip_warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1 + math.cos(math.pi * progress))


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def encode_to_embedding(model, tokens, pad_mask, cfg: Config):
    """encoder → synthesize → flatten → L2-normalize. Returns (B, N*d_sine)."""
    A, f, phi = model.encoder(tokens, pad_mask=pad_mask)
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)  # (B, N, d_sine)
    emb = signal.flatten(1)
    return F.normalize(emb, dim=-1)


def contrastive_loss(emb_a, emb_b, logit_scale):
    """Symmetric InfoNCE on a (B, B) similarity matrix."""
    scale = logit_scale.exp()
    logits = (emb_a @ emb_b.T) * scale  # (B, B)
    targets = torch.arange(emb_a.size(0), device=emb_a.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss, logits


@torch.no_grad()
def validate(model, loader, logit_scale, device, cfg: Config, max_batches: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ta, ma, tp, mp = [x.to(device) for x in batch]
        ea = encode_to_embedding(model, ta, ma, cfg)
        eb = encode_to_embedding(model, tp, mp, cfg)
        loss, logits = contrastive_loss(ea, eb, logit_scale)
        targets = torch.arange(ea.size(0), device=device)
        total_loss += loss.item() * ea.size(0)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += ea.size(0)
    model.train()
    return total_loss / max(1, total), total_correct / max(1, total)


def train(cfg: Config):
    from data_clip import make_clip_loaders

    device = pick_device(cfg.device)
    print(f"[train_clip] device={device}")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_clip_loaders(cfg)
    print(
        f"[train_clip] train pairs={len(train_loader.dataset)} "
        f"val pairs={len(val_loader.dataset)}"
    )

    model = SpectralAE(cfg).to(device)
    # Decoder gets no gradient (no path from contrastive loss); exclude it from
    # the optimizer to keep things explicit and avoid wasted state.
    encoder_params = list(model.encoder.parameters())
    logit_scale = nn.Parameter(
        torch.tensor(cfg.clip_logit_scale_init, device=device, dtype=torch.float32)
    )
    enc_n = sum(p.numel() for p in encoder_params)
    print(f"[train_clip] encoder params={enc_n/1e6:.2f}M  logit_scale init={logit_scale.exp().item():.2f}")

    opt = AdamW(encoder_params + [logit_scale], lr=cfg.clip_lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, cfg))

    os.makedirs(cfg.clip_ckpt_dir, exist_ok=True)
    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    train_iter = iter(train_loader)
    while step < cfg.clip_max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        ta, ma, tp, mp = [x.to(device) for x in batch]

        ea = encode_to_embedding(model, ta, ma, cfg)
        eb = encode_to_embedding(model, tp, mp, cfg)
        loss, logits = contrastive_loss(ea, eb, logit_scale)
        if not torch.isfinite(loss):
            print(f"step {step+1:6d}: non-finite loss ({loss.item()}); skipping batch")
            opt.zero_grad(set_to_none=True)
            step += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_params + [logit_scale], cfg.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            logit_scale.clamp_(max=cfg.clip_logit_scale_max)

        targets = torch.arange(ea.size(0), device=device)
        running_loss += loss.item() * ea.size(0)
        running_correct += (logits.argmax(-1) == targets).sum().item()
        running_total += ea.size(0)
        step += 1

        if step % cfg.clip_log_every == 0:
            avg_loss = running_loss / running_total
            acc = running_correct / running_total
            dt = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            scale = logit_scale.exp().item()
            print(
                f"step {step:6d} | loss {avg_loss:7.4f} | acc {acc*100:5.2f}% | "
                f"scale {scale:6.2f} | lr {lr_now:.2e} | "
                f"{dt/cfg.clip_log_every*1000:.0f}ms/step"
            )
            running_loss = 0.0
            running_correct = running_total = 0
            t0 = time.time()

        if step % cfg.clip_val_every == 0:
            vloss, vacc = validate(model, val_loader, logit_scale, device, cfg, cfg.clip_val_batches)
            print(f"           val loss {vloss:.4f} | val acc {vacc*100:.2f}%")

        if step % cfg.clip_ckpt_every == 0:
            ckpt_path = os.path.join(cfg.clip_ckpt_dir, f"step_{step}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "logit_scale": logit_scale.detach().cpu(),
                    "cfg": cfg.__dict__,
                    "step": step,
                },
                ckpt_path,
            )
            print(f"           saved {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.batch_size:
        cfg.clip_batch_size = args.batch_size
    if args.max_steps:
        cfg.clip_max_steps = args.max_steps
    if args.max_len:
        cfg.clip_max_len = args.max_len

    train(cfg)


if __name__ == "__main__":
    main()
