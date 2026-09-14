# Future experiments: a scalar semantic waveform

## Starting point

The anchored encoder synthesizes six frequency-localized channels and can
combine them into one observable waveform:

```text
s(t) = (1 / sqrt(d_sine)) * sum_c s_c(t)
```

The anchored-v3 post-hoc experiment found effectively no loss from this
projection. Top-1 retrieval was 80.48% with concatenated channels and 80.47%
after summation; mean STS Spearman was 48.22 versus 48.21. Ideal fixed-bandpass
filters recovered the constituent channels from the scalar waveform at 0.9965
mean cosine. Full results and the trained-sum ablation plan are in
[DESIGN.md](./DESIGN.md).

This suggests a useful working model: the scalar signal is a
frequency-division-multiplexed semantic symbol. The six synthesis channels are
potentially internal sub-bands rather than six representation coordinates that
must remain externally visible.

There are two important limits on that claim:

1. A sampled waveform is still a vector. The present `N=2048` signal has 2,048
   scalar dimensions, so being single-channel does not by itself make it more
   compact than a conventional embedding.
2. Contrastive training establishes a similarity geometry. It does not yet
   establish that arbitrary signal operations have predictable semantic
   meanings, or that the waveform can be decoded back into text.

The experiments below are intended to distinguish a useful structured signal
from an unusual but otherwise ordinary coordinate system.

## What the structure could enable

### Robust semantic communication

A scalar waveform can be subjected to ordinary signal-channel operations:
quantization, resampling, noise, clipping, missing samples, bandwidth limits,
and lossy codecs. If semantic quality degrades gracefully, the representation
could act as a robust transmission format rather than only an in-memory
embedding.

### Superposition and associative memory

Multiple text signals can be accumulated into one memory:

```text
memory(t) = (1 / sqrt(K)) * sum_k signal_k(t)
```

The inner product is linear, so correlating a query with this memory aggregates
its match against every stored item. If useful membership information survives
as `K` grows, one waveform could represent a set, document collection, or
short-term associative memory. This is one of the clearest opportunities for
the representation to do something a plain normalized embedding is not
normally asked to do.

### Semantic filtering and frequency-localized roles

The fixed anchors provide stable regions that can be isolated with bandpass
filters. Band ablations and probes can reveal whether bands are complementary,
redundant, or dead. With additional supervision, particular regions might
eventually specialize in topic, entities, syntax, sentiment, relations, or
other semantic factors. No such disentanglement should be assumed from the
current InfoNCE objective alone.

### Progressive and bandwidth-adaptive retrieval

Retrieval might work from a prefix of the waveform, a lower sample rate, or a
limited frequency range, with additional samples or bandwidth refining the
answer. This would support coarse-to-fine search and variable-cost inference.

### Semantic signal algebra

Interpolation, addition, subtraction, convolution, time shifts, phase shifts,
and modulation are all well-defined operations on the waveform. Controlled
tests can determine whether any of them correspond to stable semantic
operations. The desirable result is not merely that a transformed waveform
has a nearest neighbor, but that the transformation behaves consistently
across held-out concepts and sentence templates.

### Reconstruction through a scalar bottleneck

A waveform decoder could be trained to recover text, a canonical paraphrase,
or structured semantic attributes. This would turn the signal into a genuine
communication bottleneck. The present contrastive checkpoints were not trained
for reconstruction, so this is a later experiment rather than an expected
property of the current signal.

### Cross-modal wave symbols

Image, audio, or other encoders could be trained to synthesize signals in the
same space. A shared scalar signal format could then support cross-modal
retrieval, superposition, and signal-level fusion.

### Sonification and signal-native implementations

The current frequency range and sample grid make the representation directly
viewable and potentially audible after amplitude normalization. Sonification
may be useful for diagnostics, though there is no reason yet to expect people
to hear semantic relationships. Longer-term implementations could explore
streaming correlators, FFT-based search, or signal-processing hardware if the
robustness and superposition experiments justify them.

## Near-term experiment queue

All tests should compare against the untouched waveform on exactly the same
examples and candidate pools. Report retrieval CE/top-1, the full STS suite,
and compositionality where applicable. Also report cosine between the original
and transformed embeddings so representational distortion can be separated
from task-level degradation.

### 1. Signal-channel robustness benchmark

This is the highest-priority characterization experiment. It requires no new
training and can be run on the best multichannel-v3 checkpoint post-hoc and on
each trained summed checkpoint.

#### Quantization

- [ ] Establish float32 and float16 baselines.
- [ ] Test signed 16-, 8-, 4-, and 2-bit uniform quantization.
- [ ] Compare one global calibration scale with per-waveform scaling.
- [ ] Record storage in bytes alongside semantic performance.

The important question is whether semantic accuracy stays flat through 8 or 4
bits, not whether waveform samples can be reconstructed exactly.

#### Noise and amplitude distortion

- [ ] Add white noise at 40, 30, 20, 10, 5, and 0 dB SNR.
- [ ] Test gain changes followed by the normal embedding normalization.
- [ ] Test hard clipping at several waveform amplitude percentiles.
- [ ] Test impulsive noise as well as Gaussian noise.

#### Missing or corrupted samples

- [ ] Randomly zero 1%, 5%, 10%, 25%, and 50% of samples.
- [ ] Remove contiguous spans of the same sizes to simulate packet loss.
- [ ] Compare zero filling, linear interpolation, and no repair.

#### Resampling

- [ ] Downsample `2048 -> 1024 -> 512 -> 256 -> 128`, using proper
      anti-aliasing, then compare both direct low-resolution embeddings and
      upsampled signals.
- [ ] Repeat without anti-aliasing as a deliberate aliasing stress test.
- [ ] Test small sample-clock/rate errors.

#### Frequency filtering

- [ ] Keep each anchored band alone.
- [ ] Remove each anchored band alone.
- [ ] Sweep low-pass and high-pass cutoffs across band boundaries.
- [ ] Remove adjacent and non-adjacent band pairs.
- [ ] Compare ideal FFT masks with practical finite filters.

Useful output plots are performance-versus-distortion curves and a heat map of
which individual or paired band removals cause the largest task loss.

### 2. Superposition-capacity benchmark

- [ ] Construct mixtures containing `K = 1, 2, 4, 8, 16, 32` sentence
      signals, scaled by `1/sqrt(K)`.
- [ ] Ask whether a query sentence, paraphrase, or related sentence is present
      in the mixture among hard negative mixtures.
- [ ] Report member-versus-nonmember score distributions, ROC-AUC, recall at a
      fixed false-positive rate, and retrieval rank.
- [ ] Stratify by semantic similarity among mixture members; near-duplicates
      are likely to interfere differently from unrelated items.
- [ ] Compare waveform superposition with summing conventional embedding
      vectors under the same protocol.
- [ ] Test whether band-aware cleanup or learned readout improves capacity.

A positive result would be a capacity curve that declines gradually and beats
the conventional-vector control at equal storage or compute. A sharp collapse
at `K=2` would still be informative: summation preserves the six internal bands
but does not automatically create an associative memory over sentences.

### 3. Band specialization and redundancy

- [ ] Measure retrieval and STS using each recovered band independently.
- [ ] Measure every band pair and compare the gain against their individual
      scores.
- [ ] Track energy share, cross-band inner products, and recoverability over
      training.
- [ ] Train lightweight probes per band for topic, sentiment, length, syntax,
      named entities, and lexical overlap.
- [ ] Compare summed training with per-channel InfoNCE weights `0.1`, `0.02`,
      and `0.0` if the controlled `0.1` and pure `0.0` runs reveal a meaningful
      difference.
- [ ] Measure whether a band is independently semantic, complementary only in
      combination, redundant, or inactive.

Probe accuracy alone is not evidence of clean semantic control. Any claimed
band role should replicate across datasets and survive controls for sentence
length and vocabulary overlap.

### 4. Dimensionality and bandwidth sweep

The current signal is not yet compact. Test whether its learned structure
allows a substantially smaller public symbol.

- [ ] Train or evaluate `N = 1024, 512, 256, 128`.
- [ ] Compare the current frequency range with narrower ranges such as
      `1-300` and `1-150`, respecting the new Nyquist limit at every `N`.
- [ ] Sweep `d_sine` and anchor count while holding public sample count fixed.
- [ ] Compare raw sample storage with sparse Fourier coefficients or learned
      spectral codecs.
- [ ] Plot semantic quality against bytes, multiply-adds, and latency.
- [ ] Include standard 384- and 768-dimensional embedding baselines at the same
      numeric precision.

At float16, the current 2,048-sample symbol occupies 4,096 bytes; a
768-dimensional float16 embedding occupies 1,536 bytes. Compression claims
should therefore be based on measured bytes at matched quality, not on the
single-channel description.

### 5. Progressive retrieval

- [ ] Evaluate contiguous time prefixes of 5%, 10%, 25%, 50%, and 75%.
- [ ] Compare prefixes with uniformly spaced sample subsets of equal size.
- [ ] Evaluate incremental frequency ranges from low to high and high to low.
- [ ] Determine whether confidence is calibrated well enough to stop early.
- [ ] If post-hoc truncation fails, train with random crop/bandwidth
      augmentation and repeat.

### 6. Semantic composition and transformation

- [ ] Interpolate between sentence pairs and track nearest neighbors along the
      path.
- [ ] Test addition as conjunction or set union using controlled templates.
- [ ] Test subtraction by removing a known component from a constructed
      mixture.
- [ ] Test analogy-style offsets against conventional embedding baselines.
- [ ] Test negation, word order, role reversal, and interacting concepts; these
      are stronger controls than topic-level similarity.
- [ ] Apply time shifts, phase shifts, modulation, and convolution separately
      and measure whether their effects are consistent.

These tests should define the desired answer before inspecting nearest
neighbors. Anecdotal plausible outputs are especially easy to overinterpret.

## Later experiments

### Train for channel robustness

If post-hoc distortion exposes graceful but limited robustness, introduce
signal augmentations during contrastive training: quantization, noise,
resampling, sample dropout, and frequency masking. Apply one transformation at
a time before combining them so improvements can be attributed correctly.

### Scalar-waveform reconstruction

Attach the one-channel decoder and test progressively stronger targets:

1. Recover coarse attributes such as length, topic, or bag-of-words content.
2. Generate a semantically equivalent canonical sentence.
3. Reconstruct the original text where the bottleneck permits it.

Track retrieval and STS alongside reconstruction so token-level decoding does
not destroy the semantic geometry.

### Cross-modal alignment

Train a small image encoder to emit the same anchored scalar representation as
its caption. Begin with image-text retrieval, then test whether image and text
signals can coexist in the same superposed memory.

### Learned semantic band allocation

Only after fixed-band behavior is well characterized, experiment with explicit
objectives that allocate roles to bands: orthogonality, redundancy reduction,
task-specific heads, or routing losses. Fixed anchors should remain as the
control because unconstrained allocation can make the interpretation unstable
across runs.

## Evaluation discipline

- Keep a frozen validation manifest so dataset and batch-size changes do not
  silently change the comparison pool.
- Use the same batch size and candidate membership for retrieval comparisons;
  InfoNCE top-1 depends on the number of in-batch negatives.
- Evaluate transformations post-hoc first. Retrain with an augmentation only
  after establishing the untrained failure curve.
- Compare against conventional embedding baselines under the same storage,
  corruption, and mixture protocols.
- Report negative results. Establishing that an operation has no semantic
  interpretation is useful evidence about what the waveform structure does
  and does not provide.
- Separate signal recoverability from semantic preservation. A high waveform
  reconstruction error may be harmless, and a visually plausible signal may
  still have poor retrieval geometry.

## Recommended immediate order

1. Finish the controlled and pure trained-summation ablations.
2. Run the post-hoc robustness benchmark on the best checkpoint.
3. Run the superposition-capacity benchmark.
4. Characterize individual bands and band removals.
5. Use those results to choose between dimensionality reduction, robustness
   training, and associative-memory training as the next model change.
