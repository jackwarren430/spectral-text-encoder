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
from model import SpectralAE, freq_separation_loss, freqs_for_separation, synthesize
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


def encode_to_signal(model, tokens, pad_mask, cfg: Config):
    """encoder → synthesize. Returns (signal, f) with signal shape
    (B, N, d_sine) and f shape (B, L, d_sine). The signal is the architectural
    bottleneck; everything downstream (main embedding, per-channel embedding,
    reconstruction) is computed from it."""
    A, f, phi = model.encoder(tokens, pad_mask=pad_mask)
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    return signal, f


def signal_to_embedding(signal, cfg: Config, channel: int | None = None):
    """Flatten + L2-normalize a (slice of a) signal into an embedding.

    channel=None uses all d_sine channels (the "full" embedding); an integer
    selects one channel for the per-channel InfoNCE.

    cfg.clip_embedding_type picks the representation:
      - "time":     flatten the waveform directly (default).
      - "spectral": take |rfft(signal)| along the time axis per channel before
                    flatten. Phase-invariant; dimensionality (N//2+1) per channel.
    """
    x = signal if channel is None else signal[:, :, channel:channel + 1]
    etype = getattr(cfg, "clip_embedding_type", "time")
    if etype == "spectral":
        spec = torch.fft.rfft(x, dim=1).abs()
        return F.normalize(spec.flatten(1), dim=-1)
    if etype == "time":
        return F.normalize(x.flatten(1), dim=-1)
    raise ValueError(f"Unknown clip_embedding_type: {etype!r}")


def pool_hidden(h, pad_mask, mode: str):
    """Pool a (B, L_eff, d_model) hidden-state tensor into (B, d_model).

    pad_mask: (B, L_eff) bool with True at padding. None means all-valid.
    """
    if mode == "cls":
        return h[:, 0]
    if pad_mask is None:
        if mode == "mean_pool":
            return h.mean(dim=1)
        if mode == "max_pool":
            return h.max(dim=1).values
        raise ValueError(f"Unknown pooling mode: {mode!r}")
    keep = (~pad_mask).to(h.dtype).unsqueeze(-1)  # (B, L_eff, 1)
    if mode == "mean_pool":
        return (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
    if mode == "max_pool":
        # Push pads to -inf so they're never the argmax. Use the dtype's min
        # to stay finite across float16/bfloat16/float32.
        neg_inf = torch.finfo(h.dtype).min
        masked = h.masked_fill(pad_mask.unsqueeze(-1), neg_inf)
        return masked.max(dim=1).values
    raise ValueError(f"Unknown pooling mode: {mode!r}")


def encode_to_pooled_embedding(model, tokens, pad_mask, cfg: Config):
    """Baseline path: encoder trunk → pool → L2-normalize. Returns (B, d_model)
    L2-normalized embedding. Used in mean_pool / cls / max_pool modes."""
    h, eff_mask = model.encoder.hidden_states(tokens, pad_mask)
    vec = pool_hidden(h, eff_mask, cfg.clip_encoder_mode)
    return F.normalize(vec, dim=-1)


def encode_to_embedding(model, tokens, pad_mask, cfg: Config):
    """Convenience wrapper. Dispatches on cfg.clip_encoder_mode. Used by
    validate, infer_clip, eval_spearman. Returns (emb, f) where f is the
    encoder's frequency tensor in spectral mode and None otherwise."""
    if cfg.clip_encoder_mode == "spectral":
        signal, f = encode_to_signal(model, tokens, pad_mask, cfg)
        emb = signal_to_embedding(signal, cfg)
        return emb, f
    emb = encode_to_pooled_embedding(model, tokens, pad_mask, cfg)
    return emb, None


def reconstruction_ce_sum(model, signal, tokens, pad_mask):
    """Decoder pass: signal → token logits via tied embedding → SUM of CE
    over non-pad positions. Returns (sum_ce, n_non_pad). Splitting out the
    numerator/denominator lets GradCache normalize chunks by the full-batch
    token count rather than per-chunk counts (which would diverge under
    variable-length sentences)."""
    L = tokens.size(1)
    decoded = model.decoder(signal, L)                         # (B, L, d_model)
    logits = decoded @ model.encoder.token_emb.weight.T        # (B, L, V)
    targets = tokens.clone()
    if pad_mask is not None:
        targets = targets.masked_fill(pad_mask, -100)
    ce_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    n = (targets != -100).sum().to(ce_sum.dtype).clamp(min=1.0)
    return ce_sum, n


def _per_channel_loss(sig_a, sig_b, logit_scale, cfg: Config):
    """Average symmetric InfoNCE across channels (each channel treated as its
    own embedding via signal_to_embedding(..., channel=c))."""
    total = sig_a.new_zeros(())
    for c in range(cfg.d_sine):
        ea = signal_to_embedding(sig_a, cfg, channel=c)
        eb = signal_to_embedding(sig_b, cfg, channel=c)
        l, _ = contrastive_loss(ea, eb, logit_scale)
        total = total + l
    return total / cfg.d_sine


def contrastive_loss(emb_a, emb_b, logit_scale):
    scale = logit_scale.exp()
    logits = (emb_a @ emb_b.T) * scale
    targets = torch.arange(emb_a.size(0), device=emb_a.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss, logits


@torch.no_grad()
def validate_pooled(model, loader, logit_scale, device, cfg: Config, max_batches: int):
    """Baseline-mode validation: contrastive loss + accuracy only (no aux / pc /
    recon)."""
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ta, ma, tp, mp = [x.to(device) for x in batch]
        emb_a = encode_to_pooled_embedding(model, ta, ma, cfg)
        emb_b = encode_to_pooled_embedding(model, tp, mp, cfg)
        loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)
        bs = emb_a.size(0)
        total_ce += loss_ce.item() * bs
        targets = torch.arange(bs, device=device)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += bs
    model.train()
    n = max(1, total)
    return {
        "ce": total_ce / n,
        "aux": 0.0,
        "pc": 0.0,
        "recon": 0.0,
        "acc": total_correct / n,
    }


@torch.no_grad()
def validate(model, loader, logit_scale, device, cfg: Config, max_batches: int, min_sep: float):
    """Compute the same loss components as training (ce, aux, and pc/recon when
    enabled) plus accuracy, averaged over val batches. Returns a dict so we can
    log all of them under event="val"."""
    model.eval()
    total_ce = 0.0
    total_aux = 0.0
    total_pc = 0.0
    total_recon = 0.0
    total_correct = 0
    total = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ta, ma, tp, mp = [x.to(device) for x in batch]
        sig_a, fa = encode_to_signal(model, ta, ma, cfg)
        sig_b, fb = encode_to_signal(model, tp, mp, cfg)
        emb_a = signal_to_embedding(sig_a, cfg)
        emb_b = signal_to_embedding(sig_b, cfg)
        loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)
        aux = 0.5 * (
            freq_separation_loss(freqs_for_separation(fa, cfg), min_sep)
            + freq_separation_loss(freqs_for_separation(fb, cfg), min_sep)
        )
        bs = emb_a.size(0)
        total_ce += loss_ce.item() * bs
        total_aux += aux.item() * bs
        if cfg.clip_per_channel_lambda > 0:
            pc = _per_channel_loss(sig_a, sig_b, logit_scale, cfg)
            total_pc += pc.item() * bs
        if cfg.clip_recon_lambda > 0:
            sa, na = reconstruction_ce_sum(model, sig_a, ta, ma)
            sb, nb = reconstruction_ce_sum(model, sig_b, tp, mp)
            recon = 0.5 * (sa / na + sb / nb)
            total_recon += recon.item() * bs
        targets = torch.arange(bs, device=device)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += bs
    model.train()
    n = max(1, total)
    return {
        "ce": total_ce / n,
        "aux": total_aux / n,
        "pc": total_pc / n,
        "recon": total_recon / n,
        "acc": total_correct / n,
    }


def micro_step_direct(model, batch, logit_scale, cfg, accum, min_sep, device):
    """Single forward+backward over the full mini-batch.

    Returns (ce, aux, pc, recon, n_correct, n_total) or None on non-finite loss.
    pc / recon are 0.0 when their respective lambdas are 0."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    sig_a, fa = encode_to_signal(model, ta, ma, cfg)
    sig_b, fb = encode_to_signal(model, tp, mp, cfg)

    emb_a = signal_to_embedding(sig_a, cfg)
    emb_b = signal_to_embedding(sig_b, cfg)
    loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)

    aux = 0.5 * (
        freq_separation_loss(freqs_for_separation(fa, cfg), min_sep)
        + freq_separation_loss(freqs_for_separation(fb, cfg), min_sep)
    )
    pc = sig_a.new_zeros(())
    if cfg.clip_per_channel_lambda > 0:
        pc = _per_channel_loss(sig_a, sig_b, logit_scale, cfg)
    recon = sig_a.new_zeros(())
    if cfg.clip_recon_lambda > 0:
        sa, na = reconstruction_ce_sum(model, sig_a, ta, ma)
        sb, nb = reconstruction_ce_sum(model, sig_b, tp, mp)
        recon = 0.5 * (sa / na + sb / nb)

    total = (loss_ce
             + cfg.clip_per_channel_lambda * pc
             + cfg.freq_sep_lambda * aux
             + cfg.clip_recon_lambda * recon)
    if not torch.isfinite(total):
        return None
    (total / accum).backward()
    targets = torch.arange(emb_a.size(0), device=device)
    correct = (logits.argmax(-1) == targets).sum().item()
    return loss_ce.item(), aux.item(), float(pc.item()), float(recon.item()), correct, emb_a.size(0)


def micro_step_grad_cache(model, batch, logit_scale, cfg, accum, min_sep, device, chunk):
    """GradCache (Gao et al. 2021) with signal-level caching.

    Pass 1: forward each chunk under no_grad to collect signals. Concatenate
    into full-batch (SIG_A, SIG_B) leaves with requires_grad. Compute
    signal-derived losses (main + per-channel InfoNCE) on those leaves and
    backward — populates SIG_A.grad, SIG_B.grad, logit_scale.grad with the
    /accum factor.

    Pass 2: re-forward each chunk WITH grad. Compute chunk-local losses (aux,
    reconstruction) and combine with the cached signal gradient via a single
    torch.autograd.backward call. Aux/recon contributions are weighted by
    chunk_size/B so the average matches direct mode."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    B = ta.size(0)

    # Pass 1: signals only, no grad. Per-chunk RNG snapshot so pass 2 can
    # reproduce dropout etc.
    rng_states = []
    sigs_a, sigs_b = [], []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        rng_states.append(_snapshot_rng(device))
        with torch.no_grad():
            sa, _ = encode_to_signal(model, ta[s:e], ma[s:e], cfg)
            sb, _ = encode_to_signal(model, tp[s:e], mp[s:e], cfg)
        sigs_a.append(sa)
        sigs_b.append(sb)

    SIG_A = torch.cat(sigs_a, dim=0).detach().requires_grad_(True)
    SIG_B = torch.cat(sigs_b, dim=0).detach().requires_grad_(True)

    EA = signal_to_embedding(SIG_A, cfg)
    EB = signal_to_embedding(SIG_B, cfg)
    loss_ce, logits = contrastive_loss(EA, EB, logit_scale)
    pc = SIG_A.new_zeros(())
    if cfg.clip_per_channel_lambda > 0:
        pc = _per_channel_loss(SIG_A, SIG_B, logit_scale, cfg)
    signal_total = loss_ce + cfg.clip_per_channel_lambda * pc
    if not torch.isfinite(signal_total):
        return None
    (signal_total / accum).backward()
    cached_dSIG_A = SIG_A.grad.detach()
    cached_dSIG_B = SIG_B.grad.detach()
    targets = torch.arange(B, device=device)
    correct = (logits.argmax(-1) == targets).sum().item()

    # For reconstruction we need full-batch non-pad counts so each chunk
    # contributes 0.5 * (chunk_ce_sum_a / n_a_full + chunk_ce_sum_b / n_b_full)
    # — these add up across chunks to exactly 0.5 * (mean_a + mean_b) (the
    # direct-mode definition), regardless of how lengths split across chunks.
    if cfg.clip_recon_lambda > 0:
        n_a_full = (~ma).sum().to(torch.float32).clamp(min=1.0)
        n_b_full = (~mp).sum().to(torch.float32).clamp(min=1.0)
    else:
        n_a_full = n_b_full = None

    # Pass 2: re-forward each chunk with grad; combine cached signal gradient
    # with chunk-local aux + reconstruction in one backward call.
    aux_weighted_sum = 0.0
    recon_total = 0.0
    for i, s in enumerate(range(0, B, chunk)):
        e = min(B, s + chunk)
        _restore_rng(rng_states[i], device)
        sig_a, fa = encode_to_signal(model, ta[s:e], ma[s:e], cfg)
        sig_b, fb = encode_to_signal(model, tp[s:e], mp[s:e], cfg)
        aux_chunk = 0.5 * (
            freq_separation_loss(freqs_for_separation(fa, cfg), min_sep)
            + freq_separation_loss(freqs_for_separation(fb, cfg), min_sep)
        )
        n_chunk = e - s
        aux_weighted_sum += aux_chunk.item() * n_chunk

        recon_chunk = sig_a.new_zeros(())
        if cfg.clip_recon_lambda > 0:
            sa, _ = reconstruction_ce_sum(model, sig_a, ta[s:e], ma[s:e])
            sb, _ = reconstruction_ce_sum(model, sig_b, tp[s:e], mp[s:e])
            recon_chunk = 0.5 * (sa / n_a_full + sb / n_b_full)
            recon_total += recon_chunk.item()

        # Aux: freq_separation_loss already averages over its batch dim, so
        # weight by n_chunk/B. Reconstruction: recon_chunk is already a
        # fractional contribution to the full-batch mean — no n_chunk weighting.
        extras = (
            cfg.freq_sep_lambda * aux_chunk * n_chunk / B
            + cfg.clip_recon_lambda * recon_chunk
        ) / accum
        torch.autograd.backward(
            tensors=[sig_a, sig_b, extras],
            grad_tensors=[cached_dSIG_A[s:e], cached_dSIG_B[s:e], torch.ones_like(extras)],
        )
    return (loss_ce.item(), aux_weighted_sum / B, float(pc.item()),
            recon_total, correct, B)


def micro_step_direct_pooled(model, batch, logit_scale, cfg, accum, device):
    """Baseline-mode direct step: forward → pool → L2-normalize → contrastive
    loss → backward. No aux / pc / recon."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    emb_a = encode_to_pooled_embedding(model, ta, ma, cfg)
    emb_b = encode_to_pooled_embedding(model, tp, mp, cfg)
    loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)
    if not torch.isfinite(loss_ce):
        return None
    (loss_ce / accum).backward()
    targets = torch.arange(emb_a.size(0), device=device)
    correct = (logits.argmax(-1) == targets).sum().item()
    return loss_ce.item(), 0.0, 0.0, 0.0, correct, emb_a.size(0)


def micro_step_grad_cache_pooled(model, batch, logit_scale, cfg, accum, device, chunk):
    """Baseline-mode GradCache. Cache the L2-normalized pooled embedding rather
    than the synthesized signal — same recipe as spectral mode but the cached
    tensor is (B, d_model) instead of (B, n_samples, d_sine), and there are no
    chunk-local aux / recon terms to recombine in pass 2."""
    ta, ma, tp, mp = [x.to(device) for x in batch]
    B = ta.size(0)

    rng_states = []
    embs_a, embs_b = [], []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        rng_states.append(_snapshot_rng(device))
        with torch.no_grad():
            ea = encode_to_pooled_embedding(model, ta[s:e], ma[s:e], cfg)
            eb = encode_to_pooled_embedding(model, tp[s:e], mp[s:e], cfg)
        embs_a.append(ea)
        embs_b.append(eb)

    EMB_A = torch.cat(embs_a, dim=0).detach().requires_grad_(True)
    EMB_B = torch.cat(embs_b, dim=0).detach().requires_grad_(True)

    loss_ce, logits = contrastive_loss(EMB_A, EMB_B, logit_scale)
    if not torch.isfinite(loss_ce):
        return None
    (loss_ce / accum).backward()
    cached_dEMB_A = EMB_A.grad.detach()
    cached_dEMB_B = EMB_B.grad.detach()
    targets = torch.arange(B, device=device)
    correct = (logits.argmax(-1) == targets).sum().item()

    for i, s in enumerate(range(0, B, chunk)):
        e = min(B, s + chunk)
        _restore_rng(rng_states[i], device)
        ea = encode_to_pooled_embedding(model, ta[s:e], ma[s:e], cfg)
        eb = encode_to_pooled_embedding(model, tp[s:e], mp[s:e], cfg)
        torch.autograd.backward(
            tensors=[ea, eb],
            grad_tensors=[cached_dEMB_A[s:e], cached_dEMB_B[s:e]],
        )
    return loss_ce.item(), 0.0, 0.0, 0.0, correct, B


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

    spectral_mode = cfg.clip_encoder_mode == "spectral"
    if not spectral_mode:
        # Baseline modes can't compute these losses (no signal, no channels).
        # Catch this at startup so a typo in a long-running config doesn't
        # silently fail later under chunked GradCache.
        bad = []
        if cfg.clip_per_channel_lambda != 0:
            bad.append(f"clip_per_channel_lambda={cfg.clip_per_channel_lambda}")
        if cfg.clip_recon_lambda != 0:
            bad.append(f"clip_recon_lambda={cfg.clip_recon_lambda}")
        if cfg.freq_sep_lambda != 0:
            bad.append(f"freq_sep_lambda={cfg.freq_sep_lambda}")
        if bad:
            raise ValueError(
                f"clip_encoder_mode={cfg.clip_encoder_mode!r} requires "
                f"per-channel/recon/freq-sep lambdas to be 0; got {', '.join(bad)}"
            )

    model = SpectralAE(cfg).to(device)
    encoder_params = list(model.encoder.parameters())
    # Decoder participates only when reconstruction aux is enabled (and only
    # exists at all in spectral mode).
    use_recon = spectral_mode and cfg.clip_recon_lambda > 0
    decoder_params = list(model.decoder.parameters()) if use_recon else []
    logit_scale = nn.Parameter(
        torch.tensor(cfg.clip_logit_scale_init, device=device, dtype=torch.float32)
    )
    enc_n = sum(p.numel() for p in encoder_params)
    dec_n = sum(p.numel() for p in decoder_params)
    eff_batch = cfg.clip_batch_size * cfg.clip_grad_accum_steps
    use_grad_cache = (
        cfg.clip_cache_chunk_size is not None
        and cfg.clip_cache_chunk_size < cfg.clip_batch_size
    )
    cache_chunk = cfg.clip_cache_chunk_size if use_grad_cache else None
    mode = f"grad_cache(chunk={cache_chunk})" if use_grad_cache else "direct"
    extras = [f"enc={cfg.clip_encoder_mode}"]
    if spectral_mode and cfg.clip_embedding_type != "time":
        extras.append(f"emb={cfg.clip_embedding_type}")
    if cfg.clip_per_channel_lambda > 0:
        extras.append(f"pc_λ={cfg.clip_per_channel_lambda}")
    if use_recon:
        extras.append(f"recon_λ={cfg.clip_recon_lambda}")
    extras_str = ("  " + " ".join(extras)) if extras else ""
    dec_str = f"  decoder params={dec_n/1e6:.2f}M" if use_recon else ""
    print(
        f"[train_clip] encoder params={enc_n/1e6:.2f}M{dec_str}  "
        f"logit_scale init={logit_scale.exp().item():.2f}  "
        f"batch={cfg.clip_batch_size}×{cfg.clip_grad_accum_steps}={eff_batch}  "
        f"mode={mode}{extras_str}"
    )

    param_groups = [{"params": encoder_params, "weight_decay": cfg.weight_decay}]
    if use_recon:
        param_groups.append({"params": decoder_params, "weight_decay": cfg.weight_decay})
    param_groups.append({"params": [logit_scale], "weight_decay": 0.0})
    opt = AdamW(param_groups, lr=cfg.clip_lr)
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
    running_pc = 0.0
    running_recon = 0.0
    running_correct = 0
    running_total = 0
    running_gnorm = 0.0
    running_gnorm_count = 0
    train_iter = iter(train_loader)
    accum = max(1, cfg.clip_grad_accum_steps)
    min_sep = cfg.freq_sep_min_bins / cfg.duration
    clipped_params = encoder_params + decoder_params + [logit_scale]

    pbar = tqdm(total=cfg.clip_max_steps, initial=step, desc="train_clip", dynamic_ncols=True)
    try:
        while step < cfg.clip_max_steps:
            opt.zero_grad(set_to_none=True)
            accum_loss_sum = 0.0
            accum_aux_sum = 0.0
            accum_pc_sum = 0.0
            accum_recon_sum = 0.0
            accum_correct = 0
            accum_total = 0
            for _ in range(accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                if spectral_mode:
                    if use_grad_cache:
                        result = micro_step_grad_cache(
                            model, batch, logit_scale, cfg, accum, min_sep, device, cache_chunk
                        )
                    else:
                        result = micro_step_direct(
                            model, batch, logit_scale, cfg, accum, min_sep, device
                        )
                else:
                    if use_grad_cache:
                        result = micro_step_grad_cache_pooled(
                            model, batch, logit_scale, cfg, accum, device, cache_chunk
                        )
                    else:
                        result = micro_step_direct_pooled(
                            model, batch, logit_scale, cfg, accum, device
                        )
                if result is None:
                    tqdm.write(f"step {step+1:6d}: non-finite loss; skipping mini-batch")
                    continue
                ce_v, aux_v, pc_v, recon_v, correct, total = result
                accum_loss_sum += ce_v * total
                accum_aux_sum += aux_v * total
                accum_pc_sum += pc_v * total
                accum_recon_sum += recon_v * total
                accum_correct += correct
                accum_total += total

            if accum_total == 0:
                step += 1
                pbar.update(1)
                continue

            gnorm = torch.nn.utils.clip_grad_norm_(clipped_params, cfg.grad_clip)
            opt.step()
            sched.step()
            with torch.no_grad():
                logit_scale.clamp_(max=cfg.clip_logit_scale_max)

            running_loss += accum_loss_sum
            running_aux += accum_aux_sum
            running_pc += accum_pc_sum
            running_recon += accum_recon_sum
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
                avg_pc = running_pc / running_total
                avg_recon = running_recon / running_total
                acc = running_correct / running_total
                dt = time.time() - t0
                ms = dt / cfg.clip_log_every * 1000
                lr_now = sched.get_last_lr()[0]
                scale = logit_scale.exp().item()
                avg_gnorm = running_gnorm / max(1, running_gnorm_count)
                extras_line = ""
                if spectral_mode:
                    extras_line += f" | aux {avg_aux:6.4f}"
                if cfg.clip_per_channel_lambda > 0:
                    extras_line += f" | pc {avg_pc:6.4f}"
                if cfg.clip_recon_lambda > 0:
                    extras_line += f" | recon {avg_recon:6.4f}"
                tqdm.write(
                    f"step {step:6d} | loss {avg_loss:7.4f}"
                    f"{extras_line} | acc {acc*100:5.2f}% | scale {scale:6.2f} | "
                    f"gnorm {avg_gnorm:6.2f} | lr {lr_now:.2e} | {ms:.0f}ms/step"
                )
                pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{acc*100:.1f}%", lr=f"{lr_now:.1e}")
                metrics.log(
                    step=step, event="train",
                    loss=f"{avg_loss:.6f}", acc=f"{acc:.6f}",
                    lr=f"{lr_now:.6e}", gnorm=f"{avg_gnorm:.4f}",
                    scale=f"{scale:.4f}", aux=f"{avg_aux:.6f}",
                    ce=f"{avg_loss:.6f}", pc=f"{avg_pc:.6f}",
                    recon=f"{avg_recon:.6f}", ms_per_step=f"{ms:.2f}",
                )
                running_loss = 0.0
                running_aux = 0.0
                running_pc = 0.0
                running_recon = 0.0
                running_correct = running_total = 0
                running_gnorm = 0.0
                running_gnorm_count = 0
                t0 = time.time()

            if step % cfg.clip_val_every == 0:
                if spectral_mode:
                    v = validate(model, val_loader, logit_scale, device, cfg,
                                 cfg.clip_val_batches, min_sep)
                else:
                    v = validate_pooled(model, val_loader, logit_scale, device, cfg,
                                        cfg.clip_val_batches)
                vextras = ""
                if spectral_mode:
                    vextras += f" | aux {v['aux']:.4f}"
                if cfg.clip_per_channel_lambda > 0:
                    vextras += f" | pc {v['pc']:.4f}"
                if cfg.clip_recon_lambda > 0:
                    vextras += f" | recon {v['recon']:.4f}"
                tqdm.write(
                    f"           val: loss {v['ce']:.4f}"
                    f"{vextras} | acc {v['acc']*100:.2f}%"
                )
                metrics.log(
                    step=step, event="val",
                    loss=f"{v['ce']:.6f}", acc=f"{v['acc']:.6f}",
                    aux=f"{v['aux']:.6f}", ce=f"{v['ce']:.6f}",
                    pc=f"{v['pc']:.6f}", recon=f"{v['recon']:.6f}",
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
