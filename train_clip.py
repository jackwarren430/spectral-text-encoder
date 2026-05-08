import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import Config
from model import SpectralAE, synthesize
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
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    emb = signal.flatten(1)
    return F.normalize(emb, dim=-1)


def contrastive_loss(emb_a, emb_b, logit_scale):
    scale = logit_scale.exp()
    logits = (emb_a @ emb_b.T) * scale
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


def train(cfg: Config, run_dir: str, resume_ckpt: str | None):
    from data_clip import make_clip_loaders

    device = pick_device(cfg.device)
    print(f"[train_clip] device={device}  run_dir={run_dir}")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_clip_loaders(cfg)
    print(
        f"[train_clip] train pairs={len(train_loader.dataset)} "
        f"val pairs={len(val_loader.dataset)}"
    )

    model = SpectralAE(cfg).to(device)
    encoder_params = list(model.encoder.parameters())
    logit_scale = nn.Parameter(
        torch.tensor(cfg.clip_logit_scale_init, device=device, dtype=torch.float32)
    )
    enc_n = sum(p.numel() for p in encoder_params)
    eff_batch = cfg.clip_batch_size * cfg.clip_grad_accum_steps
    print(
        f"[train_clip] encoder params={enc_n/1e6:.2f}M  "
        f"logit_scale init={logit_scale.exp().item():.2f}  "
        f"batch={cfg.clip_batch_size}×{cfg.clip_grad_accum_steps}={eff_batch}"
    )

    opt = AdamW(
        [
            {"params": encoder_params, "weight_decay": cfg.weight_decay},
            {"params": [logit_scale], "weight_decay": 0.0},
        ],
        lr=cfg.clip_lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, cfg))

    step = 0
    if resume_ckpt:
        blob = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(blob["model"])
        with torch.no_grad():
            logit_scale.copy_(blob["logit_scale"].to(device))
        if "opt" in blob:
            opt.load_state_dict(blob["opt"])
        if "sched" in blob:
            sched.load_state_dict(blob["sched"])
        rng_restore(blob.get("rng"))
        step = int(blob.get("step", 0))
        print(f"[train_clip] resumed from {resume_ckpt} at step {step}")

    metrics = MetricsLogger(run_dir)
    t0 = time.time()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    running_gnorm = 0.0
    running_gnorm_count = 0
    train_iter = iter(train_loader)
    accum = max(1, cfg.clip_grad_accum_steps)

    pbar = tqdm(total=cfg.clip_max_steps, initial=step, desc="train_clip", dynamic_ncols=True)
    try:
        while step < cfg.clip_max_steps:
            opt.zero_grad(set_to_none=True)
            accum_loss_sum = 0.0
            accum_correct = 0
            accum_total = 0
            for _ in range(accum):
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
                    tqdm.write(f"step {step+1:6d}: non-finite loss ({loss.item()}); skipping mini-batch")
                    continue
                (loss / accum).backward()
                targets = torch.arange(ea.size(0), device=device)
                accum_loss_sum += loss.item() * ea.size(0)
                accum_correct += (logits.argmax(-1) == targets).sum().item()
                accum_total += ea.size(0)

            if accum_total == 0:
                step += 1
                pbar.update(1)
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(encoder_params + [logit_scale], cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                logit_scale.clamp_(max=cfg.clip_logit_scale_max)

            running_loss += accum_loss_sum
            running_correct += accum_correct
            running_total += accum_total
            running_gnorm += float(gnorm)
            running_gnorm_count += 1
            step += 1
            pbar.update(1)

            if step % cfg.clip_log_every == 0:
                avg_loss = running_loss / running_total
                acc = running_correct / running_total
                dt = time.time() - t0
                ms = dt / cfg.clip_log_every * 1000
                lr_now = sched.get_last_lr()[0]
                scale = logit_scale.exp().item()
                avg_gnorm = running_gnorm / max(1, running_gnorm_count)
                tqdm.write(
                    f"step {step:6d} | loss {avg_loss:7.4f} | acc {acc*100:5.2f}% | "
                    f"scale {scale:6.2f} | gnorm {avg_gnorm:6.2f} | lr {lr_now:.2e} | "
                    f"{ms:.0f}ms/step"
                )
                pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc*100:.1f}%", lr=f"{lr_now:.1e}")
                metrics.log(
                    step=step, event="train",
                    loss=f"{avg_loss:.6f}", acc=f"{acc:.6f}",
                    lr=f"{lr_now:.6e}", gnorm=f"{avg_gnorm:.4f}",
                    scale=f"{scale:.4f}", ms_per_step=f"{ms:.2f}",
                )
                running_loss = 0.0
                running_correct = running_total = 0
                running_gnorm = 0.0
                running_gnorm_count = 0
                t0 = time.time()

            if step % cfg.clip_val_every == 0:
                vloss, vacc = validate(model, val_loader, logit_scale, device, cfg, cfg.clip_val_batches)
                tqdm.write(f"           val loss {vloss:.4f} | val acc {vacc*100:.2f}%")
                metrics.log(
                    step=step, event="val",
                    loss=f"{vloss:.6f}", acc=f"{vacc:.6f}",
                    scale=f"{logit_scale.exp().item():.4f}",
                )

            if step % cfg.clip_ckpt_every == 0:
                ckpt_path = os.path.join(run_dir, f"step_{step}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "logit_scale": logit_scale.detach().cpu(),
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
        print(f"[train_clip] plots + metrics written to {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=None, help="Run folder to resume from (uses its config.json + latest step_*.pt)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    args = p.parse_args()

    if args.resume:
        run_dir = args.resume
        cfg = load_config(run_dir)
        if args.device:
            cfg.device = args.device  # hardware availability may differ
        ignored = [k for k in ("batch_size", "accum_steps", "max_steps", "max_len") if getattr(args, k) is not None]
        if ignored:
            print(f"[train_clip] --resume: ignoring CLI overrides {ignored} (config.json is source of truth)")
        resume_ckpt = find_latest_ckpt(run_dir)
        if resume_ckpt is None:
            print(f"[train_clip] --resume: no step_*.pt in {run_dir}; starting fresh in this folder")
    else:
        cfg = Config()
        if args.device:
            cfg.device = args.device
        if args.batch_size:
            cfg.clip_batch_size = args.batch_size
        if args.accum_steps:
            cfg.clip_grad_accum_steps = args.accum_steps
        if args.max_steps:
            cfg.clip_max_steps = args.max_steps
        if args.max_len:
            cfg.clip_max_len = args.max_len
        os.makedirs(cfg.clip_ckpt_dir, exist_ok=True)
        run_dir = make_run_dir(cfg.clip_ckpt_dir)
        save_config(cfg, run_dir)
        resume_ckpt = None

    train(cfg, run_dir, resume_ckpt)


if __name__ == "__main__":
    main()
