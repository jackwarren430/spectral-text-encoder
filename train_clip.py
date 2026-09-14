import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import Config
from model import (
    SpectralAE,
    combine_signal_channels,
    freq_separation_loss,
    freqs_for_separation,
    frequency_anchor_radius,
    frequency_anchors,
    synthesize,
)
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
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _device_type(device: str) -> str:
    return str(device).split(":", 1)[0]


def _use_bf16(cfg: Config, device: str) -> bool:
    return getattr(cfg, "precision", "fp32") == "bf16" and _device_type(device) == "cuda"


def _autocast(cfg: Config, device: str):
    if not _use_bf16(cfg, device):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _move_batch(batch, device: str, cfg: Config):
    non_blocking = bool(
        _device_type(device) == "cuda" and getattr(cfg, "pin_memory", False)
    )
    return tuple(x.to(device, non_blocking=non_blocking) for x in batch)


def configure_runtime(cfg: Config, device: str) -> str:
    """Configure the CUDA math path and return the effective precision."""
    if _device_type(device) != "cuda":
        return "fp32"
    if _use_bf16(cfg, device) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision='bf16' was requested but this CUDA GPU lacks BF16 support")
    use_tf32 = bool(getattr(cfg, "cuda_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    return "bf16" if _use_bf16(cfg, device) else "fp32"


def _snapshot_rng(device):
    state = {"cpu": torch.get_rng_state()}
    if _device_type(device) == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    elif _device_type(device) == "mps" and torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if _device_type(device) == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])
    elif _device_type(device) == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _encode_to_signal_unbucketed(model, tokens, pad_mask, cfg: Config):
    A, f, phi = model.encoder(tokens, pad_mask=pad_mask)
    signal = synthesize(A, f, phi, cfg.n_samples, cfg.duration)
    return signal, f


def encode_to_signal(model, tokens, pad_mask, cfg: Config):
    """encoder → synthesize. Returns (signal, f) with signal shape
    (B, N, d_sine) and f shape (B, L, d_sine). The signal is the architectural
    bottleneck; everything downstream (main embedding, per-channel embedding,
    reconstruction) is computed from it.

    On large direct batches, rows are grouped by rounded-up real length and
    encoded at that shorter width. Batch membership and final row order are
    unchanged, so the full-batch contrastive loss sees exactly the same
    negatives while transformer and synthesis padding work is greatly reduced.
    """
    bucket = int(getattr(cfg, "clip_length_bucket_size", 0) or 0)
    if bucket <= 0 or pad_mask is None or tokens.size(0) < 2:
        return _encode_to_signal_unbucketed(model, tokens, pad_mask, cfg)

    B, L = tokens.shape
    lengths = (~pad_mask).sum(dim=-1)
    bucket_lengths = ((lengths + bucket - 1) // bucket * bucket).clamp(max=L)
    limits = torch.unique(bucket_lengths, sorted=True)
    if limits.numel() == 1 and int(limits[0]) == L:
        return _encode_to_signal_unbucketed(model, tokens, pad_mask, cfg)

    signals, freqs, row_ids = [], [], []
    for limit_t in limits:
        limit = int(limit_t)
        idx = (bucket_lengths == limit_t).nonzero(as_tuple=False).flatten()
        group_signal, group_f = _encode_to_signal_unbucketed(
            model,
            tokens.index_select(0, idx)[:, :limit],
            pad_mask.index_select(0, idx)[:, :limit],
            cfg,
        )
        signals.append(group_signal)
        # Frequency health/loss callers expect the original batch L. Values in
        # the padded tail are irrelevant because the matching valid mask is false.
        freqs.append(F.pad(group_f, (0, 0, 0, L - limit)))
        row_ids.append(idx)

    row_ids = torch.cat(row_ids)
    restore = torch.argsort(row_ids)
    signal = torch.cat(signals, dim=0).index_select(0, restore)
    f = torch.cat(freqs, dim=0).index_select(0, restore)
    assert signal.size(0) == B
    return signal, f


def signal_to_embedding(signal, cfg: Config, channel: int | None = None):
    """Flatten + L2-normalize a (slice of a) signal into an embedding.

    channel=None applies cfg.signal_channel_mode to construct the observable
    symbol; an integer always selects one pre-combination channel for the
    optional per-channel InfoNCE.

    cfg.clip_embedding_type picks the representation:
      - "time":     flatten the waveform directly (default).
      - "spectral": take |rfft(signal)| along the time axis per channel before
                    flatten. Phase-invariant; dimensionality (N//2+1) per channel.
    """
    x = (
        combine_signal_channels(signal, cfg)
        if channel is None
        else signal[:, :, channel:channel + 1]
    )
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
    signal = combine_signal_channels(signal, model.cfg)
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
    if cfg.clip_embedding_type == "spectral":
        xa = torch.fft.rfft(sig_a, dim=1).abs()
        xb = torch.fft.rfft(sig_b, dim=1).abs()
    elif cfg.clip_embedding_type == "time":
        xa, xb = sig_a, sig_b
    else:
        raise ValueError(f"Unknown clip_embedding_type: {cfg.clip_embedding_type!r}")
    # (B, N, D) -> (D, B, N), then one strided batched GEMM for every channel.
    ea = F.normalize(xa.transpose(1, 2), dim=-1).transpose(0, 1)
    eb = F.normalize(xb.transpose(1, 2), dim=-1).transpose(0, 1)
    logits = torch.bmm(ea, eb.transpose(1, 2)) * logit_scale.exp()
    D, B, _ = logits.shape
    targets = torch.arange(B, device=logits.device).repeat(D)
    ab = F.cross_entropy(logits.reshape(D * B, B), targets)
    ba = F.cross_entropy(logits.transpose(1, 2).reshape(D * B, B), targets)
    return 0.5 * (ab + ba)


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
        ta, ma, tp, mp = _move_batch(batch, device, cfg)
        with _autocast(cfg, device):
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
        "sat": 0.0,
        "mid": 0.0,
        "offset_sat": 0.0,
        "boundary": 0.0,
        "spacing": 0.0,
        "band_use": 0.0,
    }


def _freq_health(fk, valid, cfg: Config):
    """Frequency-health counters over one batch side.

    fk: (B, K) frequencies from freqs_for_separation; valid: matching real-slot
    mask (None = all real). Global mode retains its legacy sigmoid saturation
    and mid-band counters. Anchored mode instead measures local softsign
    saturation, occupancy near each allowed offset boundary, mean
    nearest-neighbor spacing within channel, and observed full-band span."""
    vals = fk[valid] if valid is not None else fk.reshape(-1)
    result = {
        "n_waves": vals.numel(),
        "n_sat": 0,
        "n_mid": 0,
        "n_offset_sat": 0,
        "n_boundary": 0,
        "spacing_sum": 0.0,
        "spacing_count": 0,
        "min_f": vals.min().item() if vals.numel() else float("inf"),
        "max_f": vals.max().item() if vals.numel() else float("-inf"),
    }
    if getattr(cfg, "frequency_param_mode", "global") == "anchored":
        D = cfg.d_sine
        if fk.size(0) % D != 0:
            raise ValueError(
                f"anchored separation rows={fk.size(0)} not divisible by d_sine={D}"
            )
        B = fk.size(0) // D
        anchors = frequency_anchors(cfg, fk.device, fk.dtype).repeat(B).unsqueeze(-1)
        ratio = (fk - anchors) / frequency_anchor_radius(cfg)
        real_ratio = ratio[valid] if valid is not None else ratio.reshape(-1)
        # softsign^-1(y) = y / (1-|y|). |raw|>4 means the local derivative
        # has fallen below 4% of its value at the anchor (|offset|>0.8 radius).
        raw = real_ratio / (1.0 - real_ratio.abs()).clamp(min=1e-7)
        result["n_offset_sat"] = (raw.abs() > 4.0).sum().item()
        result["n_boundary"] = (real_ratio.abs() > 0.9).sum().item()

        K = fk.size(1)
        if K >= 2:
            vg = torch.ones_like(fk, dtype=torch.bool) if valid is None else valid
            n_real = vg.sum(dim=-1)
            ordered = fk.masked_fill(~vg, float("inf")).sort(dim=-1).values
            gaps = ordered[:, 1:] - ordered[:, :-1]
            pair_valid = torch.arange(K - 1, device=fk.device).unsqueeze(0) < (
                n_real - 1
            ).clamp(min=0).unsqueeze(1)
            gaps = gaps.masked_fill(~pair_valid, float("inf"))
            inf = torch.full((fk.size(0), 1), float("inf"), device=fk.device, dtype=fk.dtype)
            nearest = torch.minimum(torch.cat([inf, gaps], dim=1),
                                    torch.cat([gaps, inf], dim=1))
            point_valid = torch.arange(K, device=fk.device).unsqueeze(0) < n_real.unsqueeze(1)
            point_valid &= (n_real >= 2).unsqueeze(1)
            result["spacing_sum"] = nearest.masked_fill(~point_valid, 0.0).sum().item()
            result["spacing_count"] = point_valid.sum().item()
        return result

    p = (vals - cfg.f_min) / (cfg.f_max - cfg.f_min)
    pre = torch.logit(p, eps=1e-7)
    result["n_sat"] = (pre.abs() > 4.0).sum().item()
    result["n_mid"] = (
        (vals > cfg.f_min + 20.0) & (vals < cfg.f_max - 20.0)
    ).sum().item()
    return result


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
    total_sat = total_mid = total_offset_sat = total_boundary = total_waves = 0
    spacing_sum = 0.0
    spacing_count = 0
    min_f = float("inf")
    max_f = float("-inf")
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ta, ma, tp, mp = _move_batch(batch, device, cfg)
        with _autocast(cfg, device):
            sig_a, fa = encode_to_signal(model, ta, ma, cfg)
            sig_b, fb = encode_to_signal(model, tp, mp, cfg)
            emb_a = signal_to_embedding(sig_a, cfg)
            emb_b = signal_to_embedding(sig_b, cfg)
            loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)
            fka, va = freqs_for_separation(fa, cfg, ma)
            fkb, vb = freqs_for_separation(fb, cfg, mp)
            aux = 0.5 * (
                freq_separation_loss(fka, min_sep, va)
                + freq_separation_loss(fkb, min_sep, vb)
            )
            pc = None
            if cfg.clip_per_channel_lambda > 0:
                pc = _per_channel_loss(sig_a, sig_b, logit_scale, cfg)
            recon = None
            if cfg.clip_recon_lambda > 0:
                sa, na = reconstruction_ce_sum(model, sig_a, ta, ma)
                sb, nb = reconstruction_ce_sum(model, sig_b, tp, mp)
                recon = 0.5 * (sa / na + sb / nb)
        for fk, valid in [(fka, va), (fkb, vb)]:
            health = _freq_health(fk, valid, cfg)
            total_sat += health["n_sat"]
            total_mid += health["n_mid"]
            total_offset_sat += health["n_offset_sat"]
            total_boundary += health["n_boundary"]
            total_waves += health["n_waves"]
            spacing_sum += health["spacing_sum"]
            spacing_count += health["spacing_count"]
            min_f = min(min_f, health["min_f"])
            max_f = max(max_f, health["max_f"])
        bs = emb_a.size(0)
        total_ce += loss_ce.item() * bs
        total_aux += aux.item() * bs
        if pc is not None:
            total_pc += pc.item() * bs
        if recon is not None:
            total_recon += recon.item() * bs
        targets = torch.arange(bs, device=device)
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += bs
    model.train()
    n = max(1, total)
    nw = max(1, total_waves)
    observed_span = max(0.0, max_f - min_f) if total_waves else 0.0
    return {
        "ce": total_ce / n,
        "aux": total_aux / n,
        "pc": total_pc / n,
        "recon": total_recon / n,
        "acc": total_correct / n,
        "sat": total_sat / nw,
        "mid": total_mid / nw,
        "offset_sat": total_offset_sat / nw,
        "boundary": total_boundary / nw,
        "spacing": spacing_sum / max(1, spacing_count),
        "band_use": observed_span / (cfg.f_max - cfg.f_min),
    }


def micro_step_direct(model, batch, logit_scale, cfg, accum, min_sep, device):
    """Single forward+backward over the full mini-batch.

    Returns (ce, aux, pc, recon, n_correct, n_total) or None on non-finite loss.
    pc / recon are 0.0 when their respective lambdas are 0."""
    ta, ma, tp, mp = _move_batch(batch, device, cfg)
    with _autocast(cfg, device):
        sig_a, fa = encode_to_signal(model, ta, ma, cfg)
        sig_b, fb = encode_to_signal(model, tp, mp, cfg)

        emb_a = signal_to_embedding(sig_a, cfg)
        emb_b = signal_to_embedding(sig_b, cfg)
        loss_ce, logits = contrastive_loss(emb_a, emb_b, logit_scale)

        fka, va = freqs_for_separation(fa, cfg, ma)
        fkb, vb = freqs_for_separation(fb, cfg, mp)
        aux = 0.5 * (
            freq_separation_loss(fka, min_sep, va)
            + freq_separation_loss(fkb, min_sep, vb)
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
    ta, ma, tp, mp = _move_batch(batch, device, cfg)
    B = ta.size(0)

    # Pass 1: signals only, no grad. Per-chunk RNG snapshot so pass 2 can
    # reproduce dropout etc.
    rng_states = []
    sigs_a, sigs_b = [], []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        rng_states.append(_snapshot_rng(device))
        with torch.no_grad(), _autocast(cfg, device):
            sa, _ = encode_to_signal(model, ta[s:e], ma[s:e], cfg)
            sb, _ = encode_to_signal(model, tp[s:e], mp[s:e], cfg)
        sigs_a.append(sa)
        sigs_b.append(sb)

    SIG_A = torch.cat(sigs_a, dim=0).detach().requires_grad_(True)
    SIG_B = torch.cat(sigs_b, dim=0).detach().requires_grad_(True)

    with _autocast(cfg, device):
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
        with _autocast(cfg, device):
            sig_a, fa = encode_to_signal(model, ta[s:e], ma[s:e], cfg)
            sig_b, fb = encode_to_signal(model, tp[s:e], mp[s:e], cfg)
            fka, va = freqs_for_separation(fa, cfg, ma[s:e])
            fkb, vb = freqs_for_separation(fb, cfg, mp[s:e])
            aux_chunk = 0.5 * (
                freq_separation_loss(fka, min_sep, va)
                + freq_separation_loss(fkb, min_sep, vb)
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
    ta, ma, tp, mp = _move_batch(batch, device, cfg)
    with _autocast(cfg, device):
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
    ta, ma, tp, mp = _move_batch(batch, device, cfg)
    B = ta.size(0)

    rng_states = []
    embs_a, embs_b = [], []
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        rng_states.append(_snapshot_rng(device))
        with torch.no_grad(), _autocast(cfg, device):
            ea = encode_to_pooled_embedding(model, ta[s:e], ma[s:e], cfg)
            eb = encode_to_pooled_embedding(model, tp[s:e], mp[s:e], cfg)
        embs_a.append(ea)
        embs_b.append(eb)

    EMB_A = torch.cat(embs_a, dim=0).detach().requires_grad_(True)
    EMB_B = torch.cat(embs_b, dim=0).detach().requires_grad_(True)

    with _autocast(cfg, device):
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
        with _autocast(cfg, device):
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
    precision = configure_runtime(cfg, device)
    if _device_type(device) == "cuda":
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
        capability = ".".join(map(str, torch.cuda.get_device_capability()))
        runtime = (
            f"precision={precision} tf32={bool(getattr(cfg, 'cuda_tf32', False))} "
            f"gpu={gpu} sm={capability}"
        )
    else:
        runtime = f"precision={precision}"
    print(f"[train_clip] device={device} {runtime}  run_dir={run_dir}")
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_clip_loaders(cfg, device=device)
    print(
        f"[train_clip] train pairs={len(train_loader.dataset)} "
        f"val pairs={len(val_loader.dataset)}  workers={cfg.num_workers} "
        f"pin_memory={train_loader.pin_memory}"
    )

    stsb_pairs = None
    if cfg.clip_stsb_eval:
        from transformers import AutoTokenizer
        from sts_eval import load_tokenized_sts

        stsb_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        stsb_pairs = load_tokenized_sts(
            "mteb/stsbenchmark-sts",
            "validation",
            stsb_tokenizer,
            cfg.clip_max_len,
        )
        print(
            f"[train_clip] STS-B validation pairs={len(stsb_pairs)} "
            f"batch={cfg.clip_stsb_batch_size} every={cfg.clip_val_every} steps"
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
    if spectral_mode:
        extras.append(f"freq={getattr(cfg, 'frequency_param_mode', 'global')}")
        extras.append(f"channels={getattr(cfg, 'signal_channel_mode', 'multi')}")
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
    use_fused_opt = bool(
        _device_type(device) == "cuda" and getattr(cfg, "fused_optimizer", False)
    )
    try:
        opt = AdamW(param_groups, lr=cfg.clip_lr, fused=use_fused_opt)
    except (TypeError, RuntimeError) as exc:
        if not use_fused_opt:
            raise
        print(f"[train_clip] fused AdamW unavailable ({exc}); falling back to foreach")
        opt = AdamW(param_groups, lr=cfg.clip_lr, foreach=True)
        use_fused_opt = False
    print(f"[train_clip] optimizer={'fused AdamW' if use_fused_opt else 'AdamW'}")
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
                stsb = None
                if stsb_pairs is not None:
                    from sts_eval import evaluate_tokenized_sts

                    stsb = evaluate_tokenized_sts(
                        stsb_pairs,
                        model,
                        cfg,
                        device,
                        cfg.clip_stsb_batch_size,
                        encode_to_embedding,
                        autocast_context=lambda: _autocast(cfg, device),
                    )
                vextras = ""
                if spectral_mode:
                    vextras += f" | aux {v['aux']:.4f}"
                    if getattr(cfg, "frequency_param_mode", "global") == "anchored":
                        vextras += (
                            f" | off-sat {v['offset_sat']*100:.1f}%"
                            f" | boundary {v['boundary']*100:.1f}%"
                            f" | nn {v['spacing']:.2f}Hz"
                            f" | band {v['band_use']*100:.1f}%"
                        )
                    else:
                        vextras += f" | sat {v['sat']*100:.1f}% | mid {v['mid']*100:.1f}%"
                if cfg.clip_per_channel_lambda > 0:
                    vextras += f" | pc {v['pc']:.4f}"
                if cfg.clip_recon_lambda > 0:
                    vextras += f" | recon {v['recon']:.4f}"
                if stsb is not None:
                    vextras += (
                        f" | STS-B rho {stsb['spearman']*100:.2f}"
                        f" r {stsb['pearson']*100:.2f}"
                    )
                tqdm.write(
                    f"           val: loss {v['ce']:.4f}"
                    f"{vextras} | acc {v['acc']*100:.2f}%"
                )
                metrics.log(
                    step=step, event="val",
                    loss=f"{v['ce']:.6f}", acc=f"{v['acc']:.6f}",
                    aux=f"{v['aux']:.6f}", ce=f"{v['ce']:.6f}",
                    pc=f"{v['pc']:.6f}", recon=f"{v['recon']:.6f}",
                    sat=f"{v['sat']:.6f}", mid=f"{v['mid']:.6f}",
                    offset_sat=f"{v['offset_sat']:.6f}",
                    boundary=f"{v['boundary']:.6f}",
                    spacing=f"{v['spacing']:.6f}",
                    band_use=f"{v['band_use']:.6f}",
                    stsb_spearman=(f"{stsb['spearman']:.6f}" if stsb else ""),
                    stsb_pearson=(f"{stsb['pearson']:.6f}" if stsb else ""),
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
