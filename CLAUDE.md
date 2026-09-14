# nlp_vae_memory_module

A spectral autoencoder for WikiText-103. The model is unusual — read this before diving in.

## Architecture in one paragraph

A bidirectional Transformer encoder maps a length-`L` token sequence (variable per forward pass) to `L · d_sine` sine-wave parameter triples (amplitude, frequency, phase) — each token slot emits `d_sine` independent triples. The triples are synthesized into a `d_sine`-channel waveform sampled at fixed `n_samples` over fixed `duration` (sum-of-sines per channel, summed across the `L` token slots). In the default anchored frequency mode, each output channel owns a fixed frequency region and token waves predict bounded local offsets; the separation loss is applied independently to the `L` waves that collide inside each channel. `signal_channel_mode="sum"` then forms the observable scalar waveform `sum(channels) / sqrt(d_sine)`. For reconstruction, the decoder linear-projects each observable sample to `d_model`, adds a learned positional embedding over the `n_samples`-long sequence, and an `nn.TransformerDecoder` with `L` sinusoidal positional queries (computed on the fly so any `L` works) cross-attends over those sample features. Output is projected through tied embedding weights to vocabulary logits. CLIP training activates and optimizes this decoder when `clip_recon_lambda > 0`; otherwise only the contrastive waveform path is used.

## The architectural premise (don't break this)

The information bottleneck is deliberately the summed multi-channel waveform: every bit the decoder sees has to flow through `(A, f, φ)` parameters, get synthesized into sines, and survive the per-channel sum across token slots. Do not add side channels that bypass `synthesize` — no skip connections from encoder hidden states to the decoder, no auxiliary embeddings, no extra `(A, f, φ)` parameters that aren't summed into the waveform. Capacity changes inside the encoder transformer or inside the decoder transformer are fine; widening `d_sine` is the intended knob for bottleneck width.

## Files

- `config.py` — single `Config` dataclass; all hyperparameters live here. CLIP-mode knobs live under the `clip_*` prefix.
- `data.py` — WikiText-103 loading, GPT-2 BPE tokenization, fixed `seq_len` chunking. Caches tokenized chunks to `.cache/` (slow first run, instant after).
- `data_clip.py` — multi-source sentence-pair loader (all-nli, Quora duplicates, AltLex) for CLIP-style training. Variable-length pairs are collated with padding + bool pad masks. Caches tokenized pairs to `.cache/`.
- `model.py` — `SpectralEncoder` (accepts optional `pad_mask`), `synthesize`, `WaveformDecoder`, `SpectralAE`, plus `freq_separation_loss`.
- `train.py` — autoencoder reconstruction loop. AdamW + linear-warmup-cosine, NaN-skip guard, validation, checkpointing.
- `train_clip.py` — CLIP-style contrastive loop on sentence pairs. Encodes both sides through the encoder + `synthesize`, applies the configured channel readout, L2-normalizes, and computes symmetric InfoNCE with a learnable `logit_scale`. With `clip_recon_lambda > 0`, it also reconstructs both sides and includes decoder parameters in the optimizer.
- `infer.py` — AE-mode inference. Load an AE checkpoint, run a fixed sample, print original vs. reconstructed token-by-token, plot the synthesized waveform.
- `infer_clip.py` — CLIP-mode inference. Load a CLIP checkpoint, encode two sentences, print cosine similarity (and the scaled logit), plot both waveforms overlaid per channel.

## Environment

The DGX Spark uses the project-local `.venv` described in `DGX_SPARK.md`.
System Python does not contain CUDA PyTorch. Legacy Mac work can use its
existing `dl` environment, but do not resume an FP32 checkpoint under BF16.

## Common commands

```bash
# Full reconstruction training run (auto-selects CUDA/MPS/CPU)
.venv/bin/python train.py

# Short smoke test
.venv/bin/python train.py --max-steps 100 --batch-size 16

# AE-mode inference on a reconstruction checkpoint
.venv/bin/python infer.py checkpoints/step_6000.pt

# Spark-optimized CLIP run; verify the startup banner before leaving it running
.venv/bin/python train_clip.py --device cuda

# CLIP-mode inference: similarity between two sentences from a CLIP checkpoint
.venv/bin/python infer_clip.py all-training/clip_checkpoints/step_12000.pt \
    --text-a "The cat sat on the mat." --text-b "A feline rested on the rug."
```

## Things that look broken but aren't

- **`enable_nested_tensor` warning** from `nn.TransformerEncoder` — caused by `norm_first=True` (we use pre-LN deliberately). Just disables a fast path; correctness fine.
- **High initial reconstruction CE** — a fresh full-size decoder can start in the hundreds because it is confidently wrong before the waveform carries token-discriminative signal. Judge the smoke path by finite loss/gradients and an actual decoder update, then watch the CE trend over the first training logs.
- **`aux` loss near 0 from the start** — `f_init_bias` is computed on the fly from `L` and each row's real length. In anchored mode it spreads token waves inside each channel region, so most valid within-channel pairs start far apart. Not a bug.
- **`aux` loss spiking on a single inference sample** when it was tiny during training — encoder collapsed waves on that input. If common, raise `freq_sep_lambda` or train longer.

## Known foot-guns

- **Don't compute differentiable expressions before `torch.where`-ing them.** PyTorch's autograd flows gradients through *both* branches; preserve the "double-where" pattern anywhere new division-by-near-zero appears.
- **`f_max` must stay below Nyquist.** Nyquist = `n_samples / (2 * duration)`. Currently `n_samples=2048`, `duration=1.0` → Nyquist 1024 Hz, `f_max=960` leaves 64 Hz margin. Aliasing in the synthesized waveform is silent, so respect this.
- **`seq_len` is data-only.** The model is variable-length: positional encoding (encoder), per-(slot, channel) frequency bias, and decoder queries are all computed on the fly from `L` (sinusoidal PE for the embeddings, linspace for the f-bias). `cfg.seq_len` only controls data chunking. CLIP batches support mixed lengths through right-padding masks and internal execution bucketing.
- **`n_samples` is hardcoded into the decoder's `mem_pos_emb`; `d_sine` is hardcoded into the encoder head and decoder input projection.** Changing either requires fresh training or surgery on a checkpoint.
- **Checkpoints store the full `cfg` dict.** Loaders rehydrate through `config_from_snapshot`, which preserves the legacy global-sigmoid interpretation when `frequency_param_mode` is absent. AE checkpoints (`train.py`) and CLIP checkpoints (`train_clip.py`) share the model state-dict format but CLIP checkpoints additionally store `logit_scale`. Old FFT-era checkpoints will NOT load (architecture changed). Checkpoints across different `d_sine` will NOT load each other (encoder head + decoder input proj shapes change).
- **Cross-attention cost scales with `n_samples`.** The decoder's cross-attn is `seq_len × n_samples` per head per layer. Doubling `n_samples` ~doubles decoder time and memory.

## Editing principles

- The architecture is novel and load-bearing. Before changing `synthesize` or the `(A, f, φ)` channel, or adding any path from encoder to decoder that doesn't go through `synthesize`, think about whether you're widening or weakening the bottleneck and confirm the intent with the user.
- Keep `model.py` flat — no class hierarchies, no abstractions for hypothetical future variants.
