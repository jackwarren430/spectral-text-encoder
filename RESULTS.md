# Results

State of the project as of 2026-07-10. Covers the best CLIP-trained spectral models, the
compositionality (additivity) experiments, STS Spearman evaluations, and the comparison
against conventional pooled-embedding baselines. All Spearman/Pearson numbers are ×100
(the standard convention). STS evals were run 2026-07-10 with `eval_spearman.py`
(which now forces `sine_param_mode="independent"` for checkpoints that predate that field).

## TL;DR

- The spectral method **works**: a sum-of-sines waveform used directly as a sentence
  embedding reaches **62.3 Spearman on STS-B** (50.2 avg across STS12–16 + STS-B + SICK-R),
  trained only with in-batch InfoNCE on ~547k NLI/Quora/Altlex pairs.
- Conventional pooling heads on the *same* BERT-base trunk score higher (STS-B ~69–70,
  avg ~61–63), so the waveform bottleneck currently costs ~7–8 STS-B points — the price
  of forcing everything through `L·d_sine` (A, f, φ) triples.
- Embeddings are **directionally additive but not literally additive**:
  `wave(a) + wave(b)` points the same way as `wave("a b")` (cosine 0.92–0.98) but differs
  substantially in magnitude/detail (relative L2 0.4–1.2). An early, high per-channel-loss
  model is far more additive than the fully-trained best model.
- Capacity scales cleanly with `d_sine` (the intended bottleneck knob): val retrieval
  accuracy 66% → 76% → 78% for d_sine 2 → 4 → 6 at BERT-base scale.
- The June `sine_param_mode="shared"` experiments (one shared (f, φ) per token, d_sine
  amplitudes — a single multi-dimensional signal instead of d_sine independent sines) were
  **AE-reconstruction runs (`train.py`), all stopped early or stalled**. The longest run
  (d_sine=512, n_samples=512) plateaued at exactly unigram-entropy CE — the decoder learned
  token frequencies and got nothing from the waveform — but its config also violates
  Nyquist (`f_max=960` vs 256 Hz), so it isn't a clean verdict. The two Nyquist-safe shared
  runs were killed at 50 and 1400 steps. Shared mode is **untested, not refuted** — and it
  has never been tried in CLIP mode at all.

## Best model — `all-training/runpod-breakthrough/good-run/step_38000.pt`

The "breakthrough" RunPod run (2026-05-12), still the best spectral checkpoint.

| | |
|---|---|
| Architecture | encoder d_model 512, 8 layers, 8 heads, FFN 2048 (~51.5M params); `d_sine=6`, `n_samples=2048`, independent (A, f, φ) triples |
| Embedding | flattened waveform, 2048 × 6 = 12,288 dims, L2-normalized (`clip_embedding_type="time"`) |
| Training | symmetric InfoNCE, batch 512, lr 1.5e-4, per-channel InfoNCE λ=0.1, freq-sep λ=0.05, ~39k steps on all-nli + quora-duplicates + altlex (547k pairs) |
| Val retrieval | **80.3%** top-1 in-batch accuracy @ batch 512 (train acc ~92%) |

STS suite (Spearman / Pearson):

| dataset | Spearman | Pearson |
|---|---|---|
| STS12 | 42.66 | 45.15 |
| STS13 | 36.83 | 35.11 |
| STS14 | 37.56 | 38.35 |
| STS15 | 50.62 | 45.97 |
| STS16 | 59.66 | 55.02 |
| **STS-B (test)** | **62.27** | 62.60 |
| SICK-R | 61.78 | 73.19 |
| **average** | **50.20** | 50.77 |

For calibration: random embeddings sit near 0, unsupervised GloVe-mean ≈ 40–55 avg, and
contrastively trained BERT-scale baselines (SimCSE-class) sit ≈ 76–82 avg. A 6-channel
waveform bottleneck holding 50 avg / 62 STS-B is the core "this actually works" result.

## Two other notable models

### 1. `pace-ice-runs/.../comparison-test/bert_arch_dsine_2_low_lr/.../step_26000.pt` — the extreme bottleneck

BERT-base trunk (768d / 12L / 12H, FFN 3072) but only **2 sine channels** — every token
contributes just 6 scalars (2 × (A, f, φ)) to the summed waveform, and the whole sentence
must survive as a 2-channel signal.

- Val retrieval 72.0% @ batch 512 (26k steps; the lr 3e-4 twin was unstable, this 1.5e-4
  version kept climbing).
- **STS-B Spearman 56.7** — it retains ~91% of the best model's STS-B score with **one third**
  of the bottleneck width. Together with the d_sine sweep (below) this is the cleanest
  evidence that `d_sine` behaves like a real information-capacity knob rather than just a
  parameter count.

### 2. `all-training/runpod-2/step_4000.pt` — the (nearly) additive encoder

Deeper spectral encoder (512d / 12L, `d_sine=8`) trained with a much stronger per-channel
InfoNCE weight (λ=0.5 vs 0.1) and freq-sep λ=0.1, only 5.7k steps in (val acc 56.6%).

- **STS-B Spearman 37.1** — semantically much weaker than the best model.
- But it is by far the most *compositional* checkpoint measured: `wave(a)+wave(b)` lands
  ~2× closer to `wave("a b")` than the best model manages (rel. L2 ≈ 0.42 vs ≈ 0.85, see
  next section), and its per-channel plots show clean, low-frequency, near-superposable
  waves. This is the existence proof for the "embeddings you can add like signals" goal,
  and it suggests additivity is trainable (heavier per-channel loss, earlier in training)
  but currently trades off against raw STS quality.

Honorable mention: `pace-ice-runs/.../comparison-test/bert_arch_dsine_6/.../step_22000.pt`
(BERT-base trunk, d_sine=6) — STS avg 48.9 / STS-B 60.7, i.e. **scaling the trunk from 51M to
110M params bought nothing**. The bottleneck, not the encoder, is what limits quality —
consistent with the architectural premise.

## Compositionality: is `wave(a) + wave(b) ≈ wave("a b")`?

`compositionality_test.py` encodes each sentence of a pair separately, sums the two
waveforms, and compares against the waveform of the concatenated sentence, over 10 pairs
in three categories (paraphrase-related, unrelated, interacting/coreferent). Outputs live
in `experiments/spectral_compositionality_*/`.

| model | category | avg rel. L2 ↓ | avg cosine ↑ |
|---|---|---|---|
| best model (step 38000) | related | 0.849 | 0.962 |
| | unrelated | 0.769 | 0.945 |
| | interacting | 0.916 | 0.950 |
| runpod-2 (step 4000) | related | 0.459 | 0.917 |
| | unrelated | 0.397 | 0.947 |
| | interacting | 0.410 | 0.926 |

Readings:

- **Direction is preserved, detail is not.** Cosine similarity between the sum and the
  joint waveform is 0.92–0.98 everywhere, but the residual carries 40–125% of the joint
  signal's energy. Adding embeddings gets you "about the right meaning region," not the
  exact embedding of the concatenation.
- **Context-dependence shows up where it should.** For the mature model, *interacting*
  pairs (coreference/causal: "The temperature dropped sharply." / "Everyone reached for
  warm clothes.") diverge most (rel. L2 up to 1.25) — the encoder's cross-sentence
  attention genuinely changes the waves when the sentences interact. Unrelated pairs are
  the most additive, as superposition would predict.
- **Training toward retrieval erodes additivity.** The early/high-per-channel-λ runpod-2
  checkpoint is roughly twice as additive as the fully trained best model across every
  category. The per-channel plots make this visible: runpod-2's channels are smooth
  near-single sines that overlay cleanly; the best model's high-frequency channels
  (ch 3–5) show large amplitude mismatches between sum and joint.

## Comparison to conventional embedding methods

Controlled comparison on the **same BERT-base trunk, same data, same InfoNCE loss** —
the only change is the head: spectral waveform vs standard pooling (`clip_encoder_mode`
∈ mean_pool / max_pool / cls). Baselines trained at batch 1024 (a *harder* in-batch
retrieval task) and converged in far fewer steps.

| model (head) | emb. dims | steps | val acc | STS-B ρ | STS avg ρ |
|---|---|---|---|---|---|
| max-pool | 768 | 6k | 89.0% @1024 | **70.38** | **63.09** |
| mean-pool | 768 | 6k | 88.9% @1024 | 69.07 | 62.07 |
| CLS token | 768 | 2k | 87.7% @1024 | 67.69 | 60.72 |
| spectral d_sine=6 (BERT trunk) | 12,288 | 22k | 78.1% @512 | 60.73 | 48.93 |
| spectral d_sine=4 (BERT trunk) | 8,192 | 22k | 75.7% @512 | 60.87 | — |
| spectral d_sine=2 (BERT trunk, low lr) | 4,096 | 26k | 72.0% @512 | 56.68 | — |
| **best spectral (512d/8L trunk)** | 12,288 | 38k | 80.3% @512 | 62.27 | 50.20 |

Takeaways:

- Pooling wins on raw quality by ~7–8 STS-B points / ~13 avg points, while also training
  ~4× faster. Expected: pooling reads the full 768-d hidden state; the spectral head must
  squeeze everything through per-token sine triples and a per-channel sum.
- The gap is mostly on the older STS12–14 sets; on SICK-R the spectral models actually
  match or beat the baselines (spectral d_sine=6: 63.1 vs max-pool 64.7, and it *beats*
  mean-pool's 64.9 → best spectral 61.8 is close). The bottleneck hurts fine-grained
  lexical similarity more than entailment-flavored similarity.
- What the baselines don't have: a physically structured, additive-ish, per-channel
  interpretable representation. The compositionality results above only exist for the
  spectral head.

## The larger pace-ice runs (2026-05-15 → 05-16)

What actually ran on PACE-ICE (`pace-ice-runs/all-training/`):

- **`comparison-test/bert_arch_dsine_{2,4,6}`** — d_sine sweep at BERT-base scale,
  batch 512, lr 3e-4 (plus a 1.5e-4 rerun for d_sine=2). Result: monotone capacity
  scaling (val acc 66.1% / 75.7% / 78.1% for 2/4/6). d_sine=6 at lr 3e-4 destabilized
  after ~22k steps (val acc dropped 78% → 73.7%); the surviving checkpoints are from
  before the blow-up.
- **`comparison-test/high_aux_loss{,_dsine_2}`** — freq-sep λ=0.2 + per-channel λ=0.8 at
  batch 1024. Only reached ~7k steps (66.9% / 42.1% val acc); inconclusive, but on-trend
  with "heavier auxiliary pressure slows retrieval quality" (cf. runpod-2's additivity
  trade-off).
- **`{mean,max}-pool` and `cls` baselines** — the comparison table above; each kept a
  single checkpoint (6k / 6k / 2k steps).
- **`reconstruction-loss/{05,10}_lambda`** — the recon-CE auxiliary (`clip_recon_lambda`
  0.05 / 0.1) experiments **never produced data**: all eight metrics.csv files contain
  headers only (jobs died before step 50, likely during dataset cache build). The
  recon-auxiliary question is still open.

## Open: shared-sine mode (`all-training/e2e-train/`, 2026-06-01)

`sine_param_mode="shared"` changes the representation from d_sine independent sine
channels to **one multi-dimensional signal**: each token emits a single shared (f, φ)
plus d_sine amplitudes, so its contribution to every channel is the same sine, scaled.
Five AE-reconstruction runs (`train.py`, 512d/8L encoder, batch 128) tried it on 06-01;
none got a fair shake:

| run | d_sine | n_samples | Nyquist OK? | outcome |
|---|---|---|---|---|
| 17-29-58 | 6 | 2048 | yes | aborted at step 50 (~131 s/step — impractically slow on that machine) |
| 20-35-51 | 5 | 2048 | yes | killed at step 1400: CE 58 → 13.5, token acc 2.2% |
| 23-10-46 | 512 | 1028 | no (f_max 960 > 514) | died before logging |
| 23-41-58 | 512 | 512 | no (f_max 960 > 256) | died at step 50 |
| 23-44-54 | 512 | 512 | no (f_max 960 > 256) | ran 11.8k steps; **plateaued at CE ≈ 7.5, token acc ≈ 4.4%** |

Observations, not verdicts:

- The long aliased run's CE plateau ≈ 7.5 is right at WikiText unigram entropy — the
  decoder learned token frequencies and extracted **nothing** from the waveform. With
  `f_max=960` against a 256 Hz Nyquist limit, most of the band aliases silently (the
  documented foot-gun), so this run can't condemn the shared parametrization.
- The Nyquist-safe d_sine=5 run was still above uniform CE (ln 50257 ≈ 10.8 → 13.5) at
  step 1400 but falling steadily when killed. Also note every shared run starts at CE
  250–300, vs ~22 for independent mode — the shared head's init geometry produces huge
  confident-wrong logits, which alone could explain a much slower warmup. Worth fixing
  init (or lowering `A_max`/warmup lr) before judging.
- No shared-mode **CLIP** run was ever started (the configured
  `clip_ckpt_dir=all-training/shared/test-1/` was never created).

Next steps if revisiting: shared mode, `n_samples=2048` (or `f_max ≤ 240`), fixed init,
and let it run past ~10k steps; then the same in CLIP mode.

## Reproducing the numbers

```bash
# STS suite for any CLIP checkpoint (add --all for the full 7-dataset suite)
conda run -n dl python eval_spearman.py all-training/runpod-breakthrough/good-run/step_38000.pt --all

# Compositionality test (spectral checkpoints only)
conda run -n dl python compositionality_test.py all-training/runpod-breakthrough/good-run/step_38000.pt

# Pairwise similarity sanity check
conda run -n dl python infer_clip.py all-training/runpod-breakthrough/good-run/step_38000.pt \
    --text-a "The cat sat on the mat." --text-b "A feline rested on the rug."
```

Raw eval logs from the 2026-07-10 sweep are reproduced by the commands above; training
metrics for every run are in each run directory's `metrics.csv` / `loss.png` / `acc.png`.

## Anchored and summed-channel comparison (2026-07-21)

The anchor and channel-summation experiments were reevaluated under a common retrieval
protocol: the original three validation sources (all-NLI, Quora duplicates, and AltLex)
with batches of 512. STS values are the mean across STS12–16, STS-B test, and SICK-R;
Spearman/Pearson values are reported ×100.

| checkpoint | old 3-data retrieval, B512 | STS Spearman | STS Pearson |
|---|---:|---:|---:|
| Standard anchored v3, `d_sine=6`, step 18k | 80.47% | 48.21 | 49.37 |
| Summed `d_sine=6`, step 10k | 79.41% | 50.11 | 52.14 |
| Summed `d_sine=12`, step 44k | **85.78%** | 56.98 | 59.50 |
| Summed `d_sine=12` + more data, step 22k | 84.46% | **58.28** | **60.60** |

Key conclusions:

- **Increasing `d_sine` from 6 to 12 produced the clearest improvement.** The mature
  summed-d12 model gained 5.31 retrieval points and 8.77 mean Spearman points over
  standard anchored v3.
- **The more-data run's roughly 66% native validation accuracy is not comparable to the
  earlier 80–86% figures.** Its native validation uses all six data sources and batches
  of 1,024, creating a larger, more varied candidate pool with more hard and false
  negatives. Under the controlled old-three-source/B512 protocol it reaches 84.46%, only
  1.32 points below the old-data d12 model.
- **The additional data improves semantic generalization despite the lower native
  retrieval number.** It has the best aggregate STS result measured here: 58.28
  Spearman and 60.60 Pearson, gains of 1.30 and 1.10 over the old-data d12 model.
- **Training directly with summed channels remains viable.** The d6 summed model was
  stopped at 10k while still improving; even there, it traded about 1.06 retrieval
  points for gains of 1.90 Spearman and 2.77 Pearson over standard v3. The d12 results
  show that the scalar summed waveform can exceed the multichannel v3 baseline rather
  than merely preserve it.
- For selecting a general-purpose semantic checkpoint, STS and a fixed controlled
  retrieval evaluation are more informative than comparing native validation accuracy
  across runs with different datasets or batch sizes.

Raw evaluation artifacts:

- [`experiments/channel_sum_v3_step18000.json`](experiments/channel_sum_v3_step18000.json)
- [`experiments/channel_sum_d6sum_step10000.json`](experiments/channel_sum_d6sum_step10000.json)
- [`experiments/channel_sum_d12sum_step44000.json`](experiments/channel_sum_d12sum_step44000.json)
- [`experiments/channel_sum_d12sum_1024_moredata_step22000.json`](experiments/channel_sum_d12sum_1024_moredata_step22000.json)
