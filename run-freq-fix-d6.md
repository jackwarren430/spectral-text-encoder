# Run plan: `freq-fix-d6` (2026-07-16)

Full CLIP-mode training run: the best-known recipe (the 38k "good-run",
STS-B 62.3) with the frequency-separation loss actually working for the first
time. Goal: maximize Spearman; secondary question: does the fixed loss prevent
frequency collapse? Checkpoints land in `all-training/freq-fix-d6/`.

## Background: why this run exists

The best model (`runpod-breakthrough/good-run/step_38000.pt`) shows frequency
collapse — nearly all predicted frequencies pile up at the band edges. Two root
causes were identified on 2026-07-16 (see `diagnose_freqs.py` output below):

1. **Pad-slot free-riding (a real bug, now fixed).** The separation loss was
   computed over *all* slots including padding. Pad slots get gradient from the
   aux loss but cost nothing on InfoNCE (their amplitudes are zeroed), so the
   optimizer satisfied the hinge by spreading *phantom pad frequencies* across
   the band while real waves collapsed. Logged aux during the good-run: 0.049.
   True penalty on real waves: **0.529 — 11× worse than what training saw**.
   The per-pair normalization was also diluted ~`(L_max/len)²` for short rows.
2. **Sigmoid saturation ratchet.** `f = f_min + span·σ(raw_f + bias)`: once a
   wave drifts past |pre-sigmoid| ≈ 4 the f-gradient is attenuated 50×+, so the
   band edges act as absorbing states over a long run.

Diagnostic on the good-run (4 val batches, both pair sides, ~462k real waves):

| metric | value |
|---|---|
| real waves with \|pre-sigmoid\| > 4 (saturated) | **76.2%** |
| f in [1, 21] Hz / mid-band / [940, 960] Hz | 49.5% / 21.1% / 29.4% |
| aux, pads included (what training saw) | 0.049 |
| aux, pads masked (true value) | 0.529 |
| per-unit std of raw_f across inputs | 1.05 (f still input-sensitive pre-squash) |

A fresh init reads sat ≈ 0% / mid ≈ 100% — collapse is learned, not initial.

## Code changes shipped with this plan

- `model.py` — `freqs_for_separation(f, cfg, pad_mask)` returns `(freqs, valid)`;
  `freq_separation_loss(f, min_sep, valid)` excludes pairs touching pad slots and
  normalizes per row (each sentence weighs equally; GradCache chunking recombines
  exactly). Unmasked behavior is bit-identical to the old formula.
- `train_clip.py` — masks threaded into all three aux call sites (validate,
  direct, GradCache). Validation now also logs `sat` / `mid` (see below).
- `run_utils.py` — `sat`, `mid` columns added to `metrics.csv`. (Resuming a
  *pre-fix* run dir would append misaligned columns; fresh runs are fine.)
- `diagnose_freqs.py` — new. Run it on any checkpoint for the full picture:
  saturation histogram, f distribution, input-dependence, old-vs-fixed aux.
- `config.py` — defaults now *are* this run's config.

Verified: masked == trimmed-row loss in both sine modes; pad frequencies
provably can't move the loss; direct vs GradCache gradients match (residual
~1e-5 relative diff exists at λ=0 too — pre-existing chunked-forward float
noise); 3-step end-to-end smoke on MPS.

## Configuration (all in `config.py`, snapshotted to the run's `config.json`)

| knob | value | rationale |
|---|---|---|
| `sine_param_mode` | independent | proven; shared is untested in CLIP (separate experiment) |
| `d_sine` / `n_samples` | 6 / 2048 | best-model bottleneck; Nyquist 1024 Hz > f_max 960 |
| `clip_lr` | 1.5e-4 | 3e-4 destabilized d_sine=6 at ~22k (pace-ice); 1.5e-4 made the best model |
| `clip_max_steps` | 50,000 | good-run stopped at 38k of a 100k cosine schedule, lr still high; let the anneal complete |
| `clip_per_channel_lambda` | 0.1 | best-model value; fights channel collapse |
| `freq_sep_lambda` | 0.05 | nominal parity with the good-run — but now masked, so effective pressure on real pairs is ~11× what that run felt. **This is the experiment.** |
| `clip_recon_lambda` | 0 | unproven (all recon runs died at step <50), expensive at n_samples=2048, would confound the readout |
| `clip_embedding_type` | time | proven; spectral (\|rfft\|) is a follow-up ablation |
| data | all-nli + quora + altlex (547k pairs) | same as good-run, keeps comparison clean |
| batch | 512, `clip_cache_chunk_size=512` → direct mode | matches good-run; GradCache is a memory workaround the Spark doesn't need. Set chunk back to 32 on small-memory boxes. |

## Launching on the DGX Spark

```bash
python train_clip.py --device cuda
```

- **Pass `--device cuda` and check the printed `device=` line.** The config
  default is `mps`; on a non-Mac box that silently falls back to *CPU*.
- **Env**: the Spark is aarch64 — use NVIDIA's ARM CUDA PyTorch build (NGC
  PyTorch container or the aarch64 wheels). Also needs `transformers`,
  `datasets`, `tqdm`, `matplotlib`.
- **First-run cache build**: `.cache/*.pt` is gitignored, so the first launch
  downloads the three datasets from HF and tokenizes (~547k pairs; needs
  network; this is the phase where the old PACE-ICE recon jobs died). To skip
  it, `scp` the four `.cache/sentence-transformers_*.pt` files from the Mac.
- **Resume**: `python train_clip.py --resume all-training/freq-fix-d6/<run_dir>`
  (config.json in the run dir is the source of truth; CLI overrides ignored).

### Expected speed

| box | s/step | 50k steps | basis |
|---|---|---|---|
| M5 Mac, MPS | 15 | ~9 days | measured (smoke) |
| good-run RunPod GPU, fp32 direct | 0.51 | ~7 h | measured (metrics.csv median, steps ≥ 20k) |
| Spark, this code (fp32 eager) | ~1–2 | ~14–28 h | estimate: GB10 fp32 ≈ 31 TFLOPS is its weak mode |
| Spark + bf16 autocast (not yet implemented) | ~0.4–0.7 | ~6–10 h | estimate; ±2× — GB10 unbenchmarked by us |

Untapped software lever: padding waste. Mean all-nli length is 14 tokens but a
512-batch pads to L_max ≈ 91 → **6.4× wasted compute** on any hardware.
Length-bucketed batching would reclaim most of it but changes in-batch negative
composition — skip it for this run, consider for the next. The Spark's 128 GB
is also a *quality* lever: bf16 + GradCache would allow batch 2048–4096
negatives, likely worth more Spearman than any speed gain.

## What to watch

The val line (every 1000 steps) now prints, and `metrics.csv` records:

- **`aux`** — masked separation penalty over real waves only. The old logged
  values (~0.05) were fake; expect ~0 early, then watch whether it grows.
- **`sat`** — fraction of real waves with |pre-sigmoid| > 4. Fresh init ≈ 0%;
  the collapsed good-run reads 76%. **This is the collapse early-warning.**
- **`mid`** — fraction of real frequencies > 20 Hz from both band edges.
  Fresh ≈ 100%; collapsed good-run 21%.

Good-run val-accuracy trajectory for comparison (same data, batch, and loss —
the new run *should* track or beat it):

| step | 2k | 5k | 10k | 20k | 30k | 38k |
|---|---|---|---|---|---|---|
| val acc | 29.1% | 58.3% | 68.1% | 78.0% | 79.7% | 80.0% |

Decision rules:

- **Val acc trails the table by >2–3 pts at 10k** → the 11×-stronger separation
  pressure is the suspect; restart with `freq_sep_lambda=0.01` (restart, not
  resume — resume locks the config).
- **`sat` climbs past ~30–40% anyway** → the sigmoid ratchet dominates and the
  aux loss can't fix it; next move is the anchored parameterization
  (`f = anchor(slot) ± δ·tanh(raw_f)`), which makes collapse structurally
  impossible.
- Note: full 4 Hz separation is only feasible for sentences ≲ 40 tokens
  (6·L waves in a 959 Hz band), so aux will not reach exactly 0 — falling and
  small is the success shape, not zero.

## After the run

```bash
python eval_spearman.py all-training/freq-fix-d6/<run_dir>/step_50000.pt --all
python diagnose_freqs.py all-training/freq-fix-d6/<run_dir>/step_50000.pt
python compositionality_test.py all-training/freq-fix-d6/<run_dir>/step_50000.pt
```

Success = val acc ≥ 80.3% @512 **and** STS-B Spearman > 62.3, with sat/mid
meaningfully better than 76%/21%. Also worth checking whether better frequency
spread moves the compositionality numbers (good-run: rel. L2 ≈ 0.85).

## Follow-up queue (one variable at a time, in expected-value order)

1. `clip_recon_lambda=0.05` — the open question from results.md; forces
   token-level info into the waveform.
2. `clip_embedding_type="spectral"` — phase-invariant embeddings; natural fit
   for STS.
3. Batch 2048+ via bf16 + GradCache on the Spark — more negatives.
4. Shared-sine mode in CLIP — never tried.
5. Anchored f parameterization — if sat says the ratchet still wins.
