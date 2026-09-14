# Spectral text encoder design

## Fixed channel anchors and a combined symbol (2026-07-17)

This is the agreed next design direction after the `freq-fix-d6` runs. The
existing spectral encoder emits `d_sine` output channels, but each channel is
itself the sum of one component sinusoid per real token position:

```text
s_c(t) = sum_l A[l,c] * sin(2π * f[l,c] * t + phi[l,c])
```

Thus a length-`L` row has `L * d_sine` component waves but only `d_sine`
observable signal dimensions. The current global sigmoid lets every component
frequency move across the entire `[f_min, f_max]` range. It also applies the
separation objective after flattening all `L * d_sine` frequencies, even though
equal frequencies in different output channels do not collide: the decoder
receives those channels as distinct coordinates.

### Fixed, length-agnostic channel anchors

Use exactly `d_sine` fixed anchors, one selected directly by output-channel
index. Place them at equal intervals across the usable frequency range. Using
the midpoint of each interval keeps the outer anchors away from the hard
bounds:

```text
width     = (f_max - f_min) / d_sine
anchor[c] = f_min + (c + 0.5) * width
```

The anchors do not depend on `L`, token position, batch padding, or the lengths
of neighboring examples. Channel `c` therefore has the same spectral identity
in every sentence. Each token still predicts its own local offset within that
channel's region:

```text
f[l,c] = anchor[c] + radius * softsign(raw_f[l,c])
softsign(x) = x / (1 + abs(x))
```

`radius` is bounded by approximately half the channel interval, with a guard
margin at adjacent interval boundaries. `softsign` is preferred over another
global sigmoid or `tanh`: it remains bounded but its tail derivative decays
polynomially instead of exponentially. A direct penalty on offsets near their
allowed boundary remains an optional secondary safeguard.

The offset is essential. If all token components in channel `c` were forced to
the exact same frequency, their amplitude/phase sum would algebraically reduce
to one effective sinusoid, discarding most of the token-level capacity.

### Separate frequencies only where components actually collide

Compute frequency separation independently over the `L` real-token waves
inside each channel, then average over channels. Do not flatten channels into
one `L * d_sine` pair set:

```text
freq_sep = mean_c separation(f[real_tokens, c])
```

Cross-channel frequency reuse is valid because channels are separately
observable. This removes the artificial requirement to pack `L * d_sine`
globally unique frequencies into one band; only the `L` components summed into
the same channel need repulsion. Padding remains excluded exactly as in the
current fixed separation loss.

The first controlled run should restore `freq_sep_lambda=0.05` and change only
the parameterization/separation scope. Reusing `0.2` would confound the
structural change with stronger auxiliary pressure. Track per-channel boundary
occupancy, local-offset saturation, nearest-neighbor spacing, full-band usage,
validation retrieval, STS, and compositionality. The old global
`|pre-sigmoid| > 4` saturation metric must be replaced because it is not
meaningful for the new local parameterization.

### Open ablation: sum the channels into one combined symbol

Investigate collapsing the `d_sine` channel signal into one scalar waveform:

```text
s_combined(t) = (1 / sqrt(d_sine)) * sum_c s_c(t)
```

The motivation is that linear superposition can preserve the constituents when
their spectral regions are non-overlapping: the fixed channel bands act like
frequency-division multiplexing, and a Fourier-aware decoder could in principle
separate the bands again. The `1 / sqrt(d_sine)` factor keeps variance from
growing merely because more channels are summed. It has no effect on the
current CLIP cosine embedding after L2 normalization, but still matters for
decoder scale, reconstruction, and numerical consistency.

This is not lossless for arbitrary multichannel signals: the multichannel
representation has `n_samples * d_sine` observable values while the combined
symbol has only `n_samples`. That dimensionality argument is weaker for the
anchored model, however, because its outputs are restricted to disjoint
spectral subspaces. With exactly disjoint support, summation is injective on
that restricted signal class and ideal band-pass filters can recover the
channels. The practical question is therefore whether the learned bands are
sufficiently orthogonal over the finite sampled window. Off-grid frequencies,
spectral leakage, aliasing, and near-boundary components make the separation
approximate rather than exact.

The same distinction appears directly in the similarity function. The
multichannel inner product contains only matched-channel terms,
`sum_c <s_c, t_c>`. The inner product after summing also contains cross-band
terms `sum_{c != d} <s_c, t_d>`. Fixed non-overlapping regions should make
those extra terms small; measuring their magnitude is the cleanest test of
whether channel identity is already implicit in frequency.

Keep this as a separate ablation after the fixed-channel-anchor run:

1. Train/evaluate the anchored multichannel model as the control.
2. First do a post-hoc summed evaluation of the same checkpoint for retrieval,
   STS, compositionality, cross-band inner-product energy, and recoverability.
   This isolates the readout change and needs no decoder adaptation.
3. If the post-hoc result holds up, train a true combined-symbol model and adapt
   the decoder to a one-channel input for reconstruction experiments.
4. Treat pre-sum per-channel InfoNCE as a separate choice in that trained
   ablation. Keeping it supplies privileged multichannel supervision; removing
   it is the purer test of a scalar bottleneck.

The result will distinguish two hypotheses: channel identity carries essential
independent capacity, or the anchored spectrum can multiplex that information
into a single combined symbol without materially damaging the learned
representation.

### Pipeline support (2026-07-20)

The ablation is implemented as `signal_channel_mode`:

- `"multi"` (default) preserves the existing `(N, d_sine)` symbol and flattens
  the channels for contrastive learning.
- `"sum"` exposes `(1 / sqrt(d_sine)) * sum_c s_c(t)` as an `(N, 1)` symbol to
  the main contrastive loss and waveform decoder.

Synthesis remains multichannel internally in both modes. Frequency separation
therefore continues to operate within each anchored band, and pre-sum
per-channel InfoNCE remains independently controlled by
`clip_per_channel_lambda`. This makes `signal_channel_mode="sum"` with
`clip_per_channel_lambda=0` the pure scalar-bottleneck experiment, while a
nonzero per-channel weight is the privileged-supervision variant.

`eval_channel_sum.py` evaluates both readouts from one set of encoder forwards
on identical validation pools. It also reports the RMS cross-band dot-product
term, per-band energy allocation, and ideal fixed-bandpass recoverability.
`eval_spearman.py`, `infer_clip.py`, and `compositionality_test.py` accept
`--channel-mode sum` for post-hoc checkpoint evaluation.

### V3 post-hoc result (2026-07-20)

The best saved anchored-v3 checkpoint (`step_18000.pt`) was evaluated with both
readouts without changing any model weights. Retrieval used the same 50
validation batches of 512 candidates (25,600 examples) for both modes:

| readout | validation CE | top-1 retrieval |
|---|---:|---:|
| multichannel | 0.9577 | 80.48% |
| summed | 0.9582 | 80.47% |

The full STS12-16 + STS-B + SICK-R suite was likewise unchanged: mean Spearman
was 48.22 multichannel versus 48.21 summed, with STS-B at 62.40 versus 62.37.
The ten-pair compositionality probe was also effectively identical in every
category.

The signal diagnostics explain the equivalence. Cross-channel dot-product RMS
was only 0.81% of the matched-channel term across all retrieval pairs (0.25%
on positives), and within-sentence cross terms were 0.24% of multichannel
energy. Fixed ideal band-pass filters recovered the original channels from the
sum at 0.9965 mean cosine and 0.0784 mean relative L2 error. Energy was not
uniform but every band remained active (shares 15.4%, 14.9%, 20.1%, 12.6%,
13.3%, and 23.6%).

This is strong evidence that anchored frequency bands already encode channel
identity and can be multiplexed into one scalar waveform with negligible
readout loss. It is not yet evidence that privileged per-channel supervision
is unnecessary: this checkpoint was trained with
`clip_per_channel_lambda=0.1`. A true summed-symbol training comparison should
therefore test both `0.1` (controlled readout change) and `0` (pure scalar
bottleneck).

## Decision: proceed to trained summed-symbol ablations (2026-07-20)

The post-hoc test passes decisively enough to promote channel summation from an
open readout idea to the next controlled training experiment. The working
hypothesis is now that the six anchored channels are implementation-time
sub-bands, not six independently observable representation coordinates. The
model may synthesize them separately for bookkeeping and local losses, but its
public symbol can be one scalar waveform without sacrificing the information
learned by v3.

The next comparison should hold the anchored-v3 architecture, datasets, batch
size, optimizer, frequency range, and frequency-separation loss fixed and vary
only the following:

| run | `signal_channel_mode` | `clip_per_channel_lambda` | purpose |
|---|---|---:|---|
| control | `"multi"` | 0.1 | reproduce the established multichannel recipe |
| summed-controlled | `"sum"` | 0.1 | isolate training through the summed readout while retaining v3 supervision |
| summed-pure | `"sum"` | 0.0 | test the actual scalar bottleneck without privileged channel targets |

The summed-controlled run should come first because it changes only the main
readout. If it matches the control, the summed-pure run answers the remaining
question about per-channel InfoNCE. A small intermediate weight such as `0.02`
is only warranted if removing the auxiliary causes dead or highly imbalanced
bands; it should not be the first experiment.

Frequency separation remains per anchored channel in all three runs. That loss
prevents token components inside the same band from collapsing onto one
frequency and does not require each band to carry a complete semantic
embedding. For the summed runs, monitor main retrieval/STS together with
per-band energy share, dead-band incidence, cross-band RMS interference,
ideal-bandpass recoverability, and the usual within-band frequency-health
statistics. This distinguishes healthy complementary specialization from a
nominal scalar model that simply abandons some of its bands.
