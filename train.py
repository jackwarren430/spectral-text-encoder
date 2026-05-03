import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from config import Config
from model import SpectralAE, fft_peaks, synthesize


def lr_lambda(step, cfg: Config):
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    return 0.5 * (1 + math.cos(math.pi * progress))


def pick_device(requested: str) -> str:
    if requested == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


@torch.no_grad()
def validate(model, loader, device, max_batches: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        logits, targets = model(batch)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
        total_loss += loss.item() * targets.numel()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += targets.numel()
    model.train()
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def run_check(cfg: Config):
    """Sanity test: synthesize known waves, recover via fft_peaks."""
    device = pick_device(cfg.device)
    torch.manual_seed(0)
    B, L = 4, cfg.seq_len
    A = torch.rand(B, L, device=device) * 0.8 + 0.2
    # spread frequencies at least 2 Hz apart, well within [f_min, f_max]
    base = torch.linspace(cfg.f_min + 5, cfg.f_max - 5, L, device=device)
    f = base.unsqueeze(0).expand(B, L).contiguous()
    f = f + torch.randn_like(f) * 0.5
    f = f.clamp(cfg.f_min + 1, cfg.f_max - 1)
    phi = torch.rand(B, L, device=device) * 2 * math.pi
    # sort inputs so order matches output ordering
    f, perm = torch.sort(f, dim=-1)
    A = torch.gather(A, -1, perm)
    phi = torch.gather(phi, -1, perm)
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    A_hat, f_hat, phi_hat = fft_peaks(signal, cfg.seq_len, cfg.duration)
    df = (f_hat - f).abs()
    dA = (A_hat - A).abs() / A.abs().clamp(min=1e-6)
    # phase wrap-aware diff
    dphi = (phi_hat - phi + math.pi) % (2 * math.pi) - math.pi
    print("[check] freq mean abs err:", df.mean().item(), "max:", df.max().item())
    print("[check] amp mean rel err:", dA.mean().item(), "max:", dA.max().item())
    print("[check] phase mean abs err (rad):", dphi.abs().mean().item())


def train(cfg: Config):
    from data import make_loaders

    device = pick_device(cfg.device)
    print(f"[train] device={device}")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_loaders(cfg)
    print(f"[train] train chunks={len(train_loader.dataset)} val chunks={len(val_loader.dataset)}")

    model = SpectralAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params/1e6:.2f}M")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, cfg))

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    step = 0
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    train_iter = iter(train_loader)
    while step < cfg.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device)
        logits, targets = model(batch)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        running_loss += loss.item() * targets.numel()
        running_correct += (logits.argmax(-1) == targets).sum().item()
        running_count += targets.numel()
        step += 1

        if step % cfg.log_every == 0:
            avg_loss = running_loss / running_count
            acc = running_correct / running_count
            dt = time.time() - t0
            lr_now = sched.get_last_lr()[0]
            print(
                f"step {step:6d} | loss {avg_loss:7.4f} | acc {acc*100:5.2f}% | "
                f"lr {lr_now:.2e} | {dt/cfg.log_every*1000:.0f}ms/step"
            )
            running_loss = running_correct = running_count = 0
            t0 = time.time()

        if step % cfg.val_every == 0:
            vloss, vacc = validate(model, val_loader, device, cfg.val_batches)
            print(f"           val loss {vloss:.4f} | val acc {vacc*100:.2f}%")

        if step % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f"step_{step}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step}, ckpt_path)
            print(f"           saved {ckpt_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="run FFT round-trip sanity test and exit")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.seq_len:
        cfg.seq_len = args.seq_len
    if args.n_samples:
        cfg.n_samples = args.n_samples
        cfg.f_max = min(cfg.f_max, cfg.n_samples / (2 * cfg.duration) - 32)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.max_steps:
        cfg.max_steps = args.max_steps

    if args.check:
        run_check(cfg)
        return
    train(cfg)


if __name__ == "__main__":
    main()
