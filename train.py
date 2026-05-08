import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import Config
from model import SpectralAE
from run_utils import (
    MetricsLogger,
    find_latest_ckpt,
    load_config,
    make_run_dir,
    plot_metrics,
    rng_restore,
    rng_snapshot,
    save_config,
)


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
        logits, targets, _aux = model(batch)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
        total_loss += loss.item() * targets.numel()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total_count += targets.numel()
    model.train()
    return total_loss / max(1, total_count), total_correct / max(1, total_count)


def train(cfg: Config, run_dir: str, resume_ckpt: str | None):
    from data import make_loaders

    device = pick_device(cfg.device)
    print(f"[train] device={device}  run_dir={run_dir}")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_loaders(cfg)
    print(f"[train] train chunks={len(train_loader.dataset)} val chunks={len(val_loader.dataset)}")

    model = SpectralAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params={n_params/1e6:.2f}M")

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, cfg))

    step = 0
    if resume_ckpt:
        blob = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(blob["model"])
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        if "sched" in blob:
            sched.load_state_dict(blob["sched"])
        rng_restore(blob.get("rng"))
        step = int(blob.get("step", 0))
        print(f"[train] resumed from {resume_ckpt} at step {step}")

    metrics = MetricsLogger(run_dir)
    t0 = time.time()
    running_loss = 0.0
    running_aux = 0.0
    running_correct = 0
    running_count = 0
    train_iter = iter(train_loader)

    pbar = tqdm(total=cfg.max_steps, initial=step, desc="train", dynamic_ncols=True)
    try:
        while step < cfg.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = batch.to(device)
            logits, targets, aux = model(batch)
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(0, 1))
            loss = ce + cfg.freq_sep_lambda * aux
            if not torch.isfinite(loss):
                tqdm.write(f"step {step+1:6d}: non-finite loss ({loss.item()}); skipping batch")
                opt.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            running_loss += ce.item() * targets.numel()
            running_aux += aux.item()
            running_correct += (logits.argmax(-1) == targets).sum().item()
            running_count += targets.numel()
            step += 1
            pbar.update(1)

            if step % cfg.log_every == 0:
                avg_ce = running_loss / running_count
                avg_aux = running_aux / cfg.log_every
                acc = running_correct / running_count
                dt = time.time() - t0
                ms = dt / cfg.log_every * 1000
                lr_now = sched.get_last_lr()[0]
                tqdm.write(
                    f"step {step:6d} | ce {avg_ce:7.4f} | aux {avg_aux:6.3f} | "
                    f"acc {acc*100:5.2f}% | lr {lr_now:.2e} | {ms:.0f}ms/step"
                )
                pbar.set_postfix(ce=f"{avg_ce:.3f}", acc=f"{acc*100:.1f}%", lr=f"{lr_now:.1e}")
                metrics.log(
                    step=step, event="train",
                    loss=f"{avg_ce:.6f}", acc=f"{acc:.6f}",
                    lr=f"{lr_now:.6e}", gnorm=f"{float(gnorm):.4f}",
                    aux=f"{avg_aux:.6f}", ce=f"{avg_ce:.6f}",
                    ms_per_step=f"{ms:.2f}",
                )
                running_loss = running_aux = 0.0
                running_correct = running_count = 0
                t0 = time.time()

            if step % cfg.val_every == 0:
                vloss, vacc = validate(model, val_loader, device, cfg.val_batches)
                tqdm.write(f"           val loss {vloss:.4f} | val acc {vacc*100:.2f}%")
                metrics.log(
                    step=step, event="val",
                    loss=f"{vloss:.6f}", acc=f"{vacc:.6f}",
                )

            if step % cfg.ckpt_every == 0:
                ckpt_path = os.path.join(run_dir, f"step_{step}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "rng": rng_snapshot(),
                        "cfg": cfg.__dict__,
                        "step": step,
                    },
                    ckpt_path,
                )
                tqdm.write(f"           saved {ckpt_path}")
    finally:
        pbar.close()
        metrics.close()
        plot_metrics(run_dir)
        print(f"[train] plots + metrics written to {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=None, help="Run folder to resume from (uses its config.json + latest step_*.pt)")
    p.add_argument("--device", default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()

    if args.resume:
        run_dir = args.resume
        cfg = load_config(run_dir)
        if args.device:
            cfg.device = args.device
        ignored = [k for k in ("seq_len", "n_samples", "batch_size", "max_steps") if getattr(args, k) is not None]
        if ignored:
            print(f"[train] --resume: ignoring CLI overrides {ignored} (config.json is source of truth)")
        resume_ckpt = find_latest_ckpt(run_dir)
        if resume_ckpt is None:
            print(f"[train] --resume: no step_*.pt in {run_dir}; starting fresh in this folder")
    else:
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
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        run_dir = make_run_dir(cfg.ckpt_dir)
        save_config(cfg, run_dir)
        resume_ckpt = None

    train(cfg, run_dir, resume_ckpt)


if __name__ == "__main__":
    main()
