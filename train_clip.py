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
from model import SpectralAE, freq_separation_loss, synthesize
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


def _snapshot_rng(device):
    state = {"cpu": torch.get_rng_state()}
    if device == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    elif device == "mps" and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    elif device == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def encode_to_embedding(model, tokens, pad_mask, cfg: Config):
    """encoder → synthesize → flatten → L2-normalize.

    Returns (emb, f) where emb has shape (B, N*d_sine) and f has shape
    (B, L, d_sine). f is exposed so the caller can apply freq_separation_loss
    during training; it's safe to ignore at inference."""
    A, f, phi = model.encoder(tokens, pad_mask=pad_mask)
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    emb = F.normalize(signal.flatten(1), dim=-1)
    return emb, f


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
        ea, _ = encode_to_embedding(model, ta, ma, cfg)
        eb, _ = encode_to_embedding(model, tp, mp, cfg)
        loss, logits = contrastive_loss(ea, eb, logit_scale)
        targets = torch.arange(ea.size(0), device=device)
        total_loss += loss.item() * ea.size(0)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += ea.size(0)
    model.train()
    return total_loss / max(1, total), total_correct / max(1, total)


def micro_step_direct(model, batch, logit_scale, cfg, accum, min_sep, device):
    """Single forward+backward over the full mini-batch. Returns
    (ce_value, aux_value, n_correct, n_total) or None if loss was non-finite.
    Backward has already been called when this returns."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    ea, fa = encode_to_embedding(model, ta, ma, cfg)
    eb, fb = encode_to_embedding(model, tp, mp, cfg)
    loss_ce, logits = contrastive_loss(ea, eb, logit_scale)
    aux = 0.5 * (
        freq_separation_loss(fa.flatten(1, 2), min_sep)
        + freq_separation_loss(fb.flatten(1, 2), min_sep)
    )
    total = loss_ce + cfg.freq_sep_lambda * aux
    if not torch.isfinite(total):
        return None
    (total / accum).backward()
    targets = torch.arange(ea.size(0), device=device)
    correct = (logits.argmax(-1) == targets).sum().item()
    return loss_ce.item(), aux.item(), correct, ea.size(0)


def micro_step_grad_cache(model, batch, logit_scale, cfg, accum, min_sep, device, chunk):
    """GradCache (Gao et al. 2021). Pass 1: forward each chunk under no_grad,
    collect embeddings. Compute symmetric InfoNCE on the full batch and cache
    dL/dE_a, dL/dE_b. Pass 2: re-forward each chunk WITH grad and call
    autograd.backward(emb, grad_tensors=cached_grad) so the model receives the
    full-batch gradient. Aux is per-chunk; chunk contributions are weighted by
    chunk_size/B so the average matches what direct mode would compute."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    B = ta.size(0)

    # Pass 1: collect embeddings under no_grad. Snapshot RNG per chunk so pass 2
    # can reproduce dropout (cfg.dropout=0 today, but keep this correct).
    rng_states = []
    embs_a, embs_b = [], []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        rng_states.append(_snapshot_rng(device))
        with torch.no_grad():
            ea, _ = encode_to_embedding(model, ta[s:e], ma[s:e], cfg)
            eb, _ = encode_to_embedding(model, tp[s:e], mp[s:e], cfg)
        embs_a.append(ea)
        embs_b.append(eb)

    EA = torch.cat(embs_a, dim=0).detach().requires_grad_(True)
    EB = torch.cat(embs_b, dim=0).detach().requires_grad_(True)
    loss_ce, logits = contrastive_loss(EA, EB, logit_scale)
    if not torch.isfinite(loss_ce):
        return None
    # This populates EA.grad, EB.grad, and logit_scale.grad with the /accum
    # factor baked in — matching what direct mode does.
    (loss_ce / accum).backward()
    cached_dEA = EA.grad.detach()
    cached_dEB = EB.grad.detach()
    targets = torch.arange(B, device=device)
    correct = (logits.argmax(-1) == targets).sum().item()

    # Pass 2: re-forward each chunk WITH grad, push cached gradient + aux.
    aux_weighted_sum = 0.0
    for i, s in enumerate(range(0, B, chunk)):
        e = min(B, s + chunk)
        _restore_rng(rng_states[i], device)
        ea, fa = encode_to_embedding(model, ta[s:e], ma[s:e], cfg)
        eb, fb = encode_to_embedding(model, tp[s:e], mp[s:e], cfg)
        aux_chunk = 0.5 * (
            freq_separation_loss(fa.flatten(1, 2), min_sep)
            + freq_separation_loss(fb.flatten(1, 2), min_sep)
        )
        n_chunk = e - s
        aux_weighted_sum += aux_chunk.item() * n_chunk
        # full-batch aux mean = sum_chunks(aux_chunk * n_chunk) / B, so each
        # chunk contributes lambda * aux_chunk * (n_chunk/B) / accum.
        aux_term = (cfg.freq_sep_lambda * aux_chunk * n_chunk / B) / accum
        torch.autograd.backward(
            tensors=[ea, eb, aux_term],
            grad_tensors=[cached_dEA[s:e], cached_dEB[s:e], torch.ones_like(aux_term)],
        )
    return loss_ce.item(), aux_weighted_sum / B, correct, B


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
    use_grad_cache = (
        cfg.clip_cache_chunk_size is not None
        and cfg.clip_cache_chunk_size < cfg.clip_batch_size
    )
    cache_chunk = cfg.clip_cache_chunk_size if use_grad_cache else None
    mode = f"grad_cache(chunk={cache_chunk})" if use_grad_cache else "direct"
    print(
        f"[train_clip] encoder params={enc_n/1e6:.2f}M  "
        f"logit_scale init={logit_scale.exp().item():.2f}  "
        f"batch={cfg.clip_batch_size}×{cfg.clip_grad_accum_steps}={eff_batch}  "
        f"mode={mode}"
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
    running_aux = 0.0
    running_correct = 0
    running_total = 0
    running_gnorm = 0.0
    running_gnorm_count = 0
    train_iter = iter(train_loader)
    accum = max(1, cfg.clip_grad_accum_steps)
    min_sep = cfg.freq_sep_min_bins / cfg.duration

    pbar = tqdm(total=cfg.clip_max_steps, initial=step, desc="train_clip", dynamic_ncols=True)
    try:
        while step < cfg.clip_max_steps:
            opt.zero_grad(set_to_none=True)
            accum_loss_sum = 0.0
            accum_aux_sum = 0.0
            accum_correct = 0
            accum_total = 0
            for _ in range(accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                if use_grad_cache:
                    result = micro_step_grad_cache(
                        model, batch, logit_scale, cfg, accum, min_sep, device, cache_chunk
                    )
                else:
                    result = micro_step_direct(
                        model, batch, logit_scale, cfg, accum, min_sep, device
                    )
                if result is None:
                    tqdm.write(f"step {step+1:6d}: non-finite loss; skipping mini-batch")
                    continue
                ce_v, aux_v, correct, total = result
                accum_loss_sum += ce_v * total
                accum_aux_sum += aux_v * total
                accum_correct += correct
                accum_total += total

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
            running_aux += accum_aux_sum
            running_correct += accum_correct
            running_total += accum_total
            running_gnorm += float(gnorm)
            running_gnorm_count += 1
            step += 1
            pbar.update(1)

            if device == "mps" and step % 200 == 0:
                torch.mps.empty_cache()

            if step % cfg.clip_log_every == 0:
                avg_loss = running_loss / running_total
                avg_aux = running_aux / running_total
                acc = running_correct / running_total
                dt = time.time() - t0
                ms = dt / cfg.clip_log_every * 1000
                lr_now = sched.get_last_lr()[0]
                scale = logit_scale.exp().item()
                avg_gnorm = running_gnorm / max(1, running_gnorm_count)
                tqdm.write(
                    f"step {step:6d} | loss {avg_loss:7.4f} | aux {avg_aux:6.4f} | "
                    f"acc {acc*100:5.2f}% | scale {scale:6.2f} | gnorm {avg_gnorm:6.2f} | "
                    f"lr {lr_now:.2e} | {ms:.0f}ms/step"
                )
                pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc*100:.1f}%", lr=f"{lr_now:.1e}")
                metrics.log(
                    step=step, event="train",
                    loss=f"{avg_loss:.6f}", acc=f"{acc:.6f}",
                    lr=f"{lr_now:.6e}", gnorm=f"{avg_gnorm:.4f}",
                    scale=f"{scale:.4f}", aux=f"{avg_aux:.6f}",
                    ce=f"{avg_loss:.6f}", ms_per_step=f"{ms:.2f}",
                )
                running_loss = 0.0
                running_aux = 0.0
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
