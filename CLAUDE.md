# nlp_vae_memory_module

A spectral autoencoder for WikiText-103. The model is unusual — read this before diving in.

## Architecture in one paragraph

A bidirectional Transformer encoder maps a 64-token sequence to `seq_len · d_sine` sine-wave parameter triples (amplitude, frequency, phase) — each of the 64 token slots emits `d_sine` independent triples. The triples are synthesized into a `d_sine`-channel waveform sampled at fixed `n_samples` over fixed `duration` (sum-of-sines per channel, summed across the 64 token slots). The decoder linear-projects each sample's `d_sine` channels to `d_model`, adds a learned positional embedding over the `n_samples`-long sequence, and an `nn.TransformerDecoder` with `cfg.seq_len` learned positional queries cross-attends over those sample features. Output is projected through tied embedding weights to vocabulary logits, trained with cross-entropy plus a small frequency-separation auxiliary loss applied to all `L · d_sine` waves.

## The architectural premise (don't break this)

The information bottleneck is deliberately the summed multi-channel waveform: every bit the decoder sees has to flow through `(A, f, φ)` parameters, get synthesized into sines, and survive the per-channel sum across token slots. Do not add side channels that bypass `synthesize` — no skip connections from encoder hidden states to the decoder, no auxiliary embeddings, no extra `(A, f, φ)` parameters that aren't summed into the waveform. Capacity changes inside the encoder transformer or inside the decoder transformer are fine; widening `d_sine` is the intended knob for bottleneck width.

## Files

- `config.py` — single `Config` dataclass; all hyperparameters live here. CLIP-mode knobs live under the `clip_*` prefix.
- `data.py` — WikiText-103 loading, GPT-2 BPE tokenization, fixed `seq_len` chunking. Caches tokenized chunks to `.cache/` (slow first run, instant after).
- `data_clip.py` — `sentence-transformers/all-nli` (anchor, positive) loader for CLIP-style training. Variable-length pairs collated with padding + bool pad mask. Caches tokenized pairs to `.cache/`.
- `model.py` — `SpectralEncoder` (accepts optional `pad_mask`), `synthesize`, `WaveformDecoder`, `SpectralAE`, plus `freq_separation_loss`.
- `train.py` — autoencoder reconstruction loop. AdamW + linear-warmup-cosine, NaN-skip guard, validation, checkpointing.
- `train_clip.py` — CLIP-style contrastive loop on (anchor, positive) NLI pairs. Encodes both sides through the encoder + `synthesize`, flattens the `(N, d_sine)` waveform to a single embedding per sentence, L2-normalizes, computes symmetric InfoNCE with a learnable `logit_scale`. Decoder is unused (no gradient flows to it; excluded from optimizer).
- `infer.py` — load a checkpoint, run a fixed 64-token sample, print original vs. reconstructed token-by-token, plot the synthesized waveform.

## Environment

Run Python via the `dl` conda env: `conda run -n dl python <script>`. Don't use system `python3` / `pip`.

## Common commands

```bash
# Full reconstruction training run (defaults: MPS, batch 64, 20k steps)
conda run -n dl python train.py

# Short smoke test
conda run -n dl python train.py --max-steps 100 --batch-size 16

# Inference on a checkpoint
conda run -n dl python infer.py checkpoints/step_6000.pt

# CLIP-style contrastive training on sentence-transformers/all-nli pairs
conda run -n dl python train_clip.py
```

## Things that look broken but aren't

- **`enable_nested_tensor` warning** from `nn.TransformerEncoder` — caused by `norm_first=True` (we use pre-LN deliberately). Just disables a fast path; correctness fine.
- **High initial CE (often `2·ln(V) ≈ 22`)** — at step 0 the encoder emits near-random `(A, f, φ)` so the synthesized waveform carries almost no token-discriminative signal; the decoder converges as the encoder learns to push useful info through.
- **`aux` loss near 0 from the start** — the per-slot `f_pos_bias` linspace already spreads initial frequencies across the band, so most pairs start far apart. Not a bug.
- **`aux` loss spiking on a single inference sample** when it was tiny during training — encoder collapsed waves on that input. If common, raise `freq_sep_lambda` or train longer.

## Known foot-guns

- **Don't compute differentiable expressions before `torch.where`-ing them.** PyTorch's autograd flows gradients through *both* branches; preserve the "double-where" pattern anywhere new division-by-near-zero appears.
- **`f_max` must stay below Nyquist.** Nyquist = `n_samples / (2 * duration)`. Currently `n_samples=2048`, `duration=1.0` → Nyquist 1024 Hz, `f_max=960` leaves 64 Hz margin. Aliasing in the synthesized waveform is silent, so respect this.
- **`seq_len` is data-only.** The model is variable-length: positional encoding (encoder), per-(slot, channel) frequency bias, and decoder queries are all computed on the fly from `L` (sinusoidal PE for the embeddings, linspace for the f-bias). `cfg.seq_len` only controls data chunking. Different lengths *across* forward passes are fine; mixed lengths *within* a batch would need padding masks (not implemented).
- **`n_samples` is hardcoded into the decoder's `mem_pos_emb`; `d_sine` is hardcoded into the encoder head and decoder input projection.** Changing either requires fresh training or surgery on a checkpoint.
- **Checkpoints store the full `cfg` dict.** `infer.py` rehydrates it via `Config(**blob["cfg"])`, so checkpoints from older configs are loadable as long as the *parameter shapes* haven't changed. Old FFT-era checkpoints will NOT load (decoder shape changed).
- **Cross-attention cost scales with `n_samples`.** The decoder's cross-attn is `seq_len × n_samples` per head per layer. Doubling `n_samples` ~doubles decoder time and memory.

## Editing principles

- The architecture is novel and load-bearing. Before changing `synthesize` or the `(A, f, φ)` channel, or adding any path from encoder to decoder that doesn't go through `synthesize`, think about whether you're widening or weakening the bottleneck and confirm the intent with the user.
- Keep `model.py` flat — no class hierarchies, no abstractions for hypothetical future variants.
