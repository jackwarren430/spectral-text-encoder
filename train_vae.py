"""Standalone training pipeline for the generative spectral VAE."""

import argparse
from contextlib import nullcontext
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config_vae import VAEConfig, load_vae_config, save_vae_config
from model_vae import SpectralVAE
from run_utils import (
    MetricsLogger,
    find_latest_ckpt,
    make_run_dir,
    plot_metrics,
    rng_restore,
    rng_snapshot,
)


VAE_METRIC_FIELDS = [
    "step", "event", "loss", "ce", "nll", "ppl", "acc", "exact",
    "eos_acc", "length_acc", "sample_ce", "zero_ce", "shuffle_ce",
    "kl_global", "kl_token", "kl_global_objective", "kl_token_objective",
    "beta_global", "beta_token", "active_global", "active_token",
    "posterior_mu_abs", "posterior_std", "wave_rms", "global_energy",
    "token_energy", "prior_diversity", "prior_eos_rate", "kl_global_bands",
    "kl_token_positions", "lr", "gnorm", "ms_per_step",
]


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested.startswith("mps") and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return requested


def _device_type(device: str) -> str:
    return str(device).split(":", 1)[0]


def _autocast(cfg, device):
    if cfg.precision == "bf16" and _device_type(device) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def configure_runtime(cfg, device):
    if _device_type(device) != "cuda":
        return "fp32"
    # This machine's inherited ~/.triton cache can be root-owned. Keep VAE
    # kernel artifacts in the repository cache unless the caller supplied a
    # location explicitly.
    if "TRITON_CACHE_DIR" not in os.environ:
        triton_cache = os.path.join(os.path.dirname(__file__), ".cache", "triton")
        os.makedirs(triton_cache, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = triton_cache
    if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision='bf16' requested on a GPU without BF16 support")
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.cuda_tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.cuda_tf32)
    torch.set_float32_matmul_precision("high" if cfg.cuda_tf32 else "highest")
    return cfg.precision


def lr_lambda(step: int, cfg) -> float:
    if step < cfg.vae_warmup_steps:
        return step / max(1, cfg.vae_warmup_steps)
    progress = (step - cfg.vae_warmup_steps) / max(
        1, cfg.vae_max_steps - cfg.vae_warmup_steps
    )
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def kl_anneal_factor(step: int, cfg) -> float:
    if step < cfg.kl_reconstruction_warmup_steps:
        return 0.0
    if cfg.kl_anneal_steps <= 0:
        return 1.0
    return min(
        1.0,
        (step - cfg.kl_reconstruction_warmup_steps) / cfg.kl_anneal_steps,
    )


def diagonal_gaussian_kl(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)


def _sequence_counts(logits, tokens, pad_mask, eos_token_id):
    keep = ~pad_mask
    predictions = logits.argmax(dim=-1)
    correct = ((predictions == tokens) & keep).sum()
    exact = ((predictions == tokens) | pad_mask).all(dim=-1).sum()
    lengths = keep.sum(dim=-1)
    eos_positions = (lengths - 1).clamp(min=0)
    target_eos_predictions = predictions.gather(1, eos_positions.unsqueeze(1)).squeeze(1)
    eos_correct = (target_eos_predictions == eos_token_id).sum()
    predicted_eos = predictions == eos_token_id
    has_eos = predicted_eos.any(dim=-1)
    first_eos = predicted_eos.to(torch.int64).argmax(dim=-1)
    first_eos = torch.where(
        has_eos, first_eos, torch.full_like(first_eos, predictions.size(1))
    )
    length_correct = (first_eos == eos_positions).sum()
    return correct, exact, eos_correct, length_correct


def reconstruction_ce_sum(logits, tokens, pad_mask):
    targets = tokens.masked_fill(pad_mask, -100)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def vae_objective(output, tokens, pad_mask, cfg, anneal: float = 1.0):
    """Return a token-normalized ELBO loss and additive metric numerators."""
    keep = ~pad_mask
    token_count = keep.sum().clamp(min=1)
    batch_count = tokens.size(0)
    ce_sum = reconstruction_ce_sum(output.logits, tokens, pad_mask)

    global_kl_element = diagonal_gaussian_kl(output.mu_global, output.logvar_global)
    token_kl_element = diagonal_gaussian_kl(output.mu_token, output.logvar_token)
    global_groups = global_kl_element.sum(dim=-1)
    token_groups = token_kl_element.sum(dim=-1)
    global_kl_sum = global_groups.sum()
    token_kl_sum = (token_groups * keep).sum()

    # Free bits operate on semantic bands and token slots, respectively. They
    # cap KL pressure below the configured rate; they do not force information
    # into an otherwise unused latent.
    global_objective_sum = global_groups.clamp_min(cfg.global_free_bits).sum()
    token_objective_sum = (
        token_groups.clamp_min(cfg.token_free_bits) * keep
    ).sum()
    beta_global = cfg.beta_global * anneal
    beta_token = cfg.beta_token * anneal
    loss = (
        ce_sum
        + beta_global * global_objective_sum
        + beta_token * token_objective_sum
    ) / token_count

    correct, exact, eos_correct, length_correct = _sequence_counts(
        output.logits, tokens, pad_mask, cfg.eos_token_id
    )
    valid_token_mu = output.mu_token[keep]
    valid_token_logvar = output.logvar_token[keep]
    posterior_mu_abs_sum = output.mu_global.abs().sum() + valid_token_mu.abs().sum()
    posterior_std_sum = torch.exp(0.5 * output.logvar_global).sum()
    posterior_std_sum = posterior_std_sum + torch.exp(0.5 * valid_token_logvar).sum()
    posterior_count = output.mu_global.numel() + valid_token_mu.numel()

    global_power = output.global_spectrum.abs().square()
    token_power = output.token_spectrum.abs().square()
    stats = {
        "loss_weighted_sum": loss.detach() * token_count,
        "ce_sum": ce_sum.detach(),
        "token_count": token_count.detach(),
        "batch_count": torch.as_tensor(batch_count, device=tokens.device),
        "correct": correct.detach(),
        "exact": exact.detach(),
        "eos_correct": eos_correct.detach(),
        "length_correct": length_correct.detach(),
        "global_kl_sum": global_kl_sum.detach(),
        "token_kl_sum": token_kl_sum.detach(),
        "global_objective_sum": global_objective_sum.detach(),
        "token_objective_sum": token_objective_sum.detach(),
        "beta_global_sum": torch.as_tensor(beta_global * batch_count, device=tokens.device),
        "beta_token_sum": torch.as_tensor(beta_token * batch_count, device=tokens.device),
        "posterior_mu_abs_sum": posterior_mu_abs_sum.detach(),
        "posterior_std_sum": posterior_std_sum.detach(),
        "posterior_count": torch.as_tensor(posterior_count, device=tokens.device),
        "wave_square_sum": output.waveform.detach().square().sum(),
        "wave_count": torch.as_tensor(output.waveform.numel(), device=tokens.device),
        "global_energy_sum": global_power.detach().sum(),
        "global_energy_count": torch.as_tensor(global_power.numel(), device=tokens.device),
        "token_energy_sum": token_power.detach().sum(),
        "token_energy_count": torch.as_tensor(token_power.numel(), device=tokens.device),
        "kl_global_bands_sum": global_groups.detach().sum(dim=0),
        "kl_token_positions_sum": (token_groups.detach() * keep).sum(dim=0),
        "kl_token_positions_count": keep.sum(dim=0).detach(),
    }
    return loss, stats


class VAEAccumulator:
    """Combine variable-length batches using their exact denominators."""

    _SCALARS = (
        "loss_weighted_sum", "ce_sum", "token_count", "batch_count", "correct",
        "exact", "eos_correct", "length_correct", "global_kl_sum", "token_kl_sum",
        "global_objective_sum", "token_objective_sum", "beta_global_sum",
        "beta_token_sum", "posterior_mu_abs_sum", "posterior_std_sum",
        "posterior_count", "wave_square_sum", "wave_count", "global_energy_sum",
        "global_energy_count", "token_energy_sum", "token_energy_count",
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.values = {key: 0.0 for key in self._SCALARS}
        self.global_bands = torch.zeros(cfg.global_bands, dtype=torch.float64)
        self.token_positions = torch.zeros(cfg.vae_max_length, dtype=torch.float64)
        self.token_position_counts = torch.zeros(cfg.vae_max_length, dtype=torch.float64)

    def update(self, stats):
        # One device-to-host transfer avoids synchronizing CUDA separately for
        # every scalar diagnostic on every training micro-batch.
        scalar_values = torch.stack(
            [stats[key].detach().to(torch.float64) for key in self._SCALARS]
        ).cpu().tolist()
        for key, value in zip(self._SCALARS, scalar_values):
            self.values[key] += value
        self.global_bands += stats["kl_global_bands_sum"].detach().double().cpu()
        self.token_positions += stats["kl_token_positions_sum"].detach().double().cpu()
        self.token_position_counts += (
            stats["kl_token_positions_count"].detach().double().cpu()
        )

    def finalize(self):
        v = self.values
        tokens = max(1.0, v["token_count"])
        batches = max(1.0, v["batch_count"])
        posterior_count = max(1.0, v["posterior_count"])
        ce = v["ce_sum"] / tokens
        token_position_kl = self.token_positions / self.token_position_counts.clamp(min=1)
        return {
            "loss": v["loss_weighted_sum"] / tokens,
            "ce": ce,
            "nll": v["ce_sum"] / batches,
            "ppl": math.exp(min(20.0, ce)),
            "acc": v["correct"] / tokens,
            "exact": v["exact"] / batches,
            "eos_acc": v["eos_correct"] / batches,
            "length_acc": v["length_correct"] / batches,
            "kl_global": v["global_kl_sum"] / batches,
            "kl_token": v["token_kl_sum"] / tokens,
            "kl_global_objective": v["global_objective_sum"] / tokens,
            "kl_token_objective": v["token_objective_sum"] / tokens,
            "beta_global": v["beta_global_sum"] / batches,
            "beta_token": v["beta_token_sum"] / batches,
            "posterior_mu_abs": v["posterior_mu_abs_sum"] / posterior_count,
            "posterior_std": v["posterior_std_sum"] / posterior_count,
            "wave_rms": math.sqrt(v["wave_square_sum"] / max(1.0, v["wave_count"])),
            "global_energy": v["global_energy_sum"] / max(1.0, v["global_energy_count"]),
            "token_energy": v["token_energy_sum"] / max(1.0, v["token_energy_count"]),
            "kl_global_bands": (self.global_bands / batches).tolist(),
            "kl_token_positions": token_position_kl.tolist(),
        }


class RunningVariance:
    def __init__(self, dimensions):
        self.total = torch.zeros(dimensions, dtype=torch.float64)
        self.square_total = torch.zeros(dimensions, dtype=torch.float64)
        self.count = 0

    def update(self, values):
        values = values.detach().double().cpu().reshape(-1, self.total.numel())
        self.total += values.sum(dim=0)
        self.square_total += values.square().sum(dim=0)
        self.count += values.size(0)

    def active(self, threshold):
        if self.count < 2:
            return 0
        mean = self.total / self.count
        variance = self.square_total / self.count - mean.square()
        return int((variance > threshold).sum())


def _ce_for_ablation(logits, tokens, pad_mask):
    return float(reconstruction_ce_sum(logits, tokens, pad_mask)), int((~pad_mask).sum())


@torch.no_grad()
def validate_vae(
    model, loader, device, cfg, max_batches, anneal=1.0,
    collect_text_samples: bool = False,
):
    model.eval()
    accumulator = VAEAccumulator(cfg)
    global_variance = RunningVariance(cfg.global_bands * cfg.global_latent_per_band)
    token_variance = RunningVariance(cfg.token_latent_dim)
    sample_ce_sum = zero_ce_sum = shuffle_ce_sum = 0.0
    sample_tokens = zero_tokens = shuffle_tokens = 0
    qualitative = None

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        tokens, pad_mask = _move_batch(batch, device, cfg)
        with _autocast(cfg, device):
            output = model(tokens, pad_mask, sample=False)
            _loss, stats = vae_objective(output, tokens, pad_mask, cfg, anneal)
        accumulator.update(stats)
        global_variance.update(output.mu_global)
        token_variance.update(output.mu_token[~pad_mask])

        if cfg.vae_validate_sample_reconstruction:
            with _autocast(cfg, device):
                sampled = model(tokens, pad_mask, sample=True)
            numerator, denominator = _ce_for_ablation(sampled.logits, tokens, pad_mask)
            sample_ce_sum += numerator
            sample_tokens += denominator
        else:
            sampled = None
        if cfg.vae_validate_signal_ablations:
            with _autocast(cfg, device):
                zero_logits = model.decode_waveform(torch.zeros_like(output.waveform))
                if output.waveform.size(0) > 1:
                    shuffled_waveform = output.waveform.roll(1, dims=0)
                else:
                    shuffled_waveform = output.waveform.flip(1)
                shuffle_logits = model.decode_waveform(shuffled_waveform)
            numerator, denominator = _ce_for_ablation(zero_logits, tokens, pad_mask)
            zero_ce_sum += numerator
            zero_tokens += denominator
            numerator, denominator = _ce_for_ablation(shuffle_logits, tokens, pad_mask)
            shuffle_ce_sum += numerator
            shuffle_tokens += denominator
        if collect_text_samples and qualitative is None:
            qualitative = {
                "targets": tokens.detach().cpu(),
                "mean": output.logits.argmax(dim=-1).detach().cpu(),
                "sample": (
                    sampled.logits.argmax(dim=-1).detach().cpu()
                    if sampled is not None else None
                ),
            }

    result = accumulator.finalize()
    result["active_global"] = global_variance.active(cfg.active_unit_variance)
    result["active_token"] = token_variance.active(cfg.active_unit_variance)
    result["sample_ce"] = sample_ce_sum / max(1, sample_tokens)
    result["zero_ce"] = zero_ce_sum / max(1, zero_tokens)
    result["shuffle_ce"] = shuffle_ce_sum / max(1, shuffle_tokens)

    n_prior = int(cfg.vae_prior_samples)
    if n_prior > 0:
        with _autocast(cfg, device):
            prior_waveform = model.prior_waveform(n_prior, device=device)
            prior_tokens = model.decode_waveform(prior_waveform).argmax(dim=-1)
        eos = prior_tokens == cfg.eos_token_id
        result["prior_eos_rate"] = float(eos.any(dim=-1).float().mean())
        positions = torch.arange(cfg.vae_max_length, device=device).unsqueeze(0)
        first_eos = torch.where(
            eos.any(dim=-1), eos.to(torch.int64).argmax(dim=-1),
            torch.full((n_prior,), cfg.vae_max_length, device=device),
        )
        before_eos = positions <= first_eos.unsqueeze(1)
        generated = prior_tokens[before_eos]
        result["prior_diversity"] = (
            generated.unique().numel() / max(1, generated.numel())
        )
        if collect_text_samples:
            if qualitative is None:
                qualitative = {}
            qualitative["prior"] = prior_tokens.detach().cpu()
    else:
        result["prior_eos_rate"] = 0.0
        result["prior_diversity"] = 0.0
    if collect_text_samples:
        result["text_samples"] = qualitative
    model.train()
    return result


def _move_batch(batch, device, cfg):
    non_blocking = _device_type(device) == "cuda" and cfg.pin_memory
    return tuple(value.to(device, non_blocking=non_blocking) for value in batch)


def _format_metric_row(values):
    row = {}
    for key, value in values.items():
        if key in {"kl_global_bands", "kl_token_positions"}:
            row[key] = json.dumps([round(float(item), 6) for item in value])
        elif isinstance(value, float):
            row[key] = f"{value:.6f}"
        else:
            row[key] = value
    return row


def _decode_token_rows(tokenizer, rows, eos_token_id):
    if rows is None:
        return []
    decoded = []
    for row in rows.tolist():
        if eos_token_id in row:
            row = row[: row.index(eos_token_id)]
        decoded.append(tokenizer.decode(row))
    return decoded


def write_text_samples(run_dir, step, samples, tokenizer, cfg, limit=8):
    if not samples:
        return None
    path = os.path.join(run_dir, f"samples_step_{step}.txt")
    targets = _decode_token_rows(tokenizer, samples.get("targets"), cfg.eos_token_id)
    means = _decode_token_rows(tokenizer, samples.get("mean"), cfg.eos_token_id)
    sampled = _decode_token_rows(tokenizer, samples.get("sample"), cfg.eos_token_id)
    priors = _decode_token_rows(tokenizer, samples.get("prior"), cfg.eos_token_id)
    with open(path, "w") as handle:
        handle.write(f"spectral VAE validation samples at step {step}\n\n")
        for index in range(min(limit, len(targets))):
            handle.write(f"heldout[{index}] target: {targets[index]!r}\n")
            handle.write(f"heldout[{index}] mean:   {means[index]!r}\n")
            if index < len(sampled):
                handle.write(f"heldout[{index}] sample: {sampled[index]!r}\n")
            handle.write("\n")
        for index, text in enumerate(priors[:limit]):
            handle.write(f"prior[{index}]: {text!r}\n")
    return path


def initialize_encoder_from_clip(model, checkpoint_path, device):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = blob["model"]
    destination = model.state_dict()
    prefixes = ("encoder.token_emb.", "encoder.encoder.")
    compatible = {
        key: value
        for key, value in source.items()
        if key.startswith(prefixes)
        and key in destination
        and destination[key].shape == value.shape
    }
    if not compatible:
        raise RuntimeError(
            f"no compatible text-encoder weights found in {checkpoint_path}"
        )
    destination.update(compatible)
    model.load_state_dict(destination)
    print(
        f"[train_vae] initialized {len(compatible)} text-encoder tensors "
        f"from {checkpoint_path} (source step {blob.get('step', '?')})"
    )


def save_checkpoint(model, optimizer, scheduler, cfg, run_dir, step):
    path = os.path.join(run_dir, f"step_{step}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "rng": rng_snapshot(),
            "cfg": cfg.__dict__,
            "step": step,
            "pipeline": "spectral_vae",
        },
        path,
    )
    return path


def train(cfg, run_dir, resume_ckpt=None, init_encoder_from=None):
    from data_vae import make_vae_loaders

    device = pick_device(cfg.device)
    precision = configure_runtime(cfg, device)
    print(f"[train_vae] device={device} precision={precision} run_dir={run_dir}")
    torch.manual_seed(cfg.seed)
    if _device_type(device) == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    train_loader, val_loader = make_vae_loaders(cfg, device=device)
    print(
        f"[train_vae] train sentences={len(train_loader.dataset)} "
        f"val sentences={len(val_loader.dataset)} workers={cfg.num_workers}"
    )
    from transformers import AutoTokenizer

    sample_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    model = SpectralVAE(cfg).to(device)
    if init_encoder_from:
        initialize_encoder_from_clip(model, init_encoder_from, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        f"[train_vae] params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"trainable={sum(p.numel() for p in trainable) / 1e6:.2f}M "
        f"latent=global({cfg.global_bands}x{cfg.global_latent_per_band})+"
        f"token({cfg.vae_max_length}x{cfg.token_latent_dim})"
    )
    fused = _device_type(device) == "cuda" and cfg.fused_optimizer
    try:
        optimizer = AdamW(
            trainable, lr=cfg.vae_lr, weight_decay=cfg.weight_decay, fused=fused
        )
    except (TypeError, RuntimeError) as error:
        if not fused:
            raise
        print(f"[train_vae] fused AdamW unavailable ({error}); using foreach")
        optimizer = AdamW(
            trainable, lr=cfg.vae_lr, weight_decay=cfg.weight_decay, foreach=True
        )
        fused = False
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda current: lr_lambda(current, cfg)
    )

    step = 0
    if resume_ckpt:
        blob = torch.load(resume_ckpt, map_location=device, weights_only=False)
        if blob.get("pipeline") not in {None, "spectral_vae"}:
            raise RuntimeError(f"{resume_ckpt} is not a spectral-VAE checkpoint")
        model.load_state_dict(blob["model"])
        if "opt" in blob:
            optimizer.load_state_dict(blob["opt"])
        if "sched" in blob:
            scheduler.load_state_dict(blob["sched"])
        rng_restore(blob.get("rng"))
        step = int(blob.get("step", 0))
        print(f"[train_vae] resumed {resume_ckpt} at step {step}")

    metrics = MetricsLogger(run_dir, fields=VAE_METRIC_FIELDS)
    running = VAEAccumulator(cfg)
    running_steps = 0
    running_gnorm = 0.0
    started = time.time()
    train_iterator = iter(train_loader)
    accumulation = cfg.vae_grad_accum_steps
    pbar = tqdm(total=cfg.vae_max_steps, initial=step, desc="train_vae", dynamic_ncols=True)
    completed = False
    try:
        while step < cfg.vae_max_steps:
            optimizer.zero_grad(set_to_none=True)
            step_accumulator = VAEAccumulator(cfg)
            finite = True
            anneal = kl_anneal_factor(step, cfg)
            for _ in range(accumulation):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                tokens, pad_mask = _move_batch(batch, device, cfg)
                with _autocast(cfg, device):
                    output = model(tokens, pad_mask, sample=True)
                    loss, stats = vae_objective(output, tokens, pad_mask, cfg, anneal)
                if not torch.isfinite(loss):
                    finite = False
                    break
                (loss / accumulation).backward()
                step_accumulator.update(stats)

            if finite:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                # Fold the completed optimizer step into the log window.
                for key in running._SCALARS:
                    running.values[key] += step_accumulator.values[key]
                running.global_bands += step_accumulator.global_bands
                running.token_positions += step_accumulator.token_positions
                running.token_position_counts += step_accumulator.token_position_counts
                running_gnorm += float(gradient_norm)
                running_steps += 1
            else:
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(f"step {step + 1:6d}: non-finite ELBO; skipped")
            step += 1
            pbar.update(1)

            if step % cfg.vae_log_every == 0 and running_steps:
                train_metrics = running.finalize()
                elapsed_ms = (time.time() - started) / cfg.vae_log_every * 1000.0
                avg_gnorm = running_gnorm / running_steps
                lr = scheduler.get_last_lr()[0]
                tqdm.write(
                    f"step {step:6d} | loss {train_metrics['loss']:.4f} "
                    f"ce {train_metrics['ce']:.4f} | KL g {train_metrics['kl_global']:.3f} "
                    f"t {train_metrics['kl_token']:.3f} | acc {train_metrics['acc']*100:.2f}% "
                    f"| beta {train_metrics['beta_global']:.3f}/"
                    f"{train_metrics['beta_token']:.3f} | {elapsed_ms:.0f}ms/step"
                )
                pbar.set_postfix(
                    ce=f"{train_metrics['ce']:.3f}",
                    kl=f"{train_metrics['kl_global']:.2f}/{train_metrics['kl_token']:.2f}",
                )
                train_metrics.update(
                    step=step, event="train", lr=f"{lr:.6e}",
                    gnorm=f"{avg_gnorm:.4f}", ms_per_step=f"{elapsed_ms:.2f}",
                )
                metrics.log(**_format_metric_row(train_metrics))
                running = VAEAccumulator(cfg)
                running_steps = 0
                running_gnorm = 0.0
                started = time.time()

            if step % cfg.vae_val_every == 0:
                # Posterior/prior validation samples must not perturb the
                # stochastic training trajectory or make it depend on how
                # frequently validation is configured.
                validation_rng = rng_snapshot()
                try:
                    validation = validate_vae(
                        model, val_loader, device, cfg, cfg.vae_val_batches, anneal,
                        collect_text_samples=True,
                    )
                finally:
                    rng_restore(validation_rng)
                text_samples = validation.pop("text_samples", None)
                sample_path = write_text_samples(
                    run_dir, step, text_samples, sample_tokenizer, cfg
                )
                tqdm.write(
                    f"           val ce {validation['ce']:.4f} "
                    f"acc {validation['acc']*100:.2f}% exact {validation['exact']*100:.2f}% "
                    f"| KL g {validation['kl_global']:.3f} t {validation['kl_token']:.3f} "
                    f"| zero/shuffle {validation['zero_ce']:.3f}/"
                    f"{validation['shuffle_ce']:.3f} | active "
                    f"{validation['active_global']}/{validation['active_token']}"
                )
                validation.update(step=step, event="val")
                metrics.log(**_format_metric_row(validation))
                if sample_path:
                    tqdm.write(f"           samples {sample_path}")

            if step % cfg.vae_ckpt_every == 0:
                path = save_checkpoint(model, optimizer, scheduler, cfg, run_dir, step)
                tqdm.write(f"           saved {path}")
        completed = True
        if step and step % cfg.vae_ckpt_every:
            path = save_checkpoint(model, optimizer, scheduler, cfg, run_dir, step)
            tqdm.write(f"           saved final {path}")
    finally:
        pbar.close()
        metrics.close()
        plot_metrics(run_dir)
        status = "complete" if completed else "interrupted"
        print(f"[train_vae] {status}; plots + metrics written to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=None, help="VAE run folder to resume")
    parser.add_argument(
        "--init-encoder-from", default=None,
        help="CLIP checkpoint whose compatible text-encoder weights initialize a fresh VAE",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    if args.resume and args.init_encoder_from:
        raise SystemExit("--resume and --init-encoder-from are mutually exclusive")
    if args.resume:
        run_dir = args.resume
        cfg = load_vae_config(run_dir)
        if args.device:
            cfg.device = args.device
        ignored = [
            name for name in ("batch_size", "accum_steps", "max_steps", "max_length")
            if getattr(args, name) is not None
        ]
        if ignored:
            print(f"[train_vae] resume ignores config overrides: {ignored}")
        checkpoint = find_latest_ckpt(run_dir)
        if checkpoint is None:
            print(f"[train_vae] no checkpoint in {run_dir}; starting at step zero")
    else:
        cfg = VAEConfig()
        if args.device:
            cfg.device = args.device
        if args.batch_size:
            cfg.vae_batch_size = args.batch_size
        if args.accum_steps:
            cfg.vae_grad_accum_steps = args.accum_steps
        if args.max_steps:
            cfg.vae_max_steps = args.max_steps
        if args.max_length:
            cfg.vae_max_length = args.max_length
        # Re-run invariants after mutable CLI overrides.
        cfg.__post_init__()
        os.makedirs(cfg.vae_ckpt_dir, exist_ok=True)
        run_dir = make_run_dir(cfg.vae_ckpt_dir)
        save_vae_config(cfg, run_dir)
        checkpoint = None
    train(cfg, run_dir, checkpoint, init_encoder_from=args.init_encoder_from)


if __name__ == "__main__":
    main()
