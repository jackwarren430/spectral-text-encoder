# Run plan: `freq-fix-d6-v3` (2026-07-17)

This is the controlled fixed-channel-anchor run described in `DESIGN.md`.
It keeps the six-channel waveform and the proven CLIP recipe, but removes the
global sigmoid's band-edge failure mode and stops penalizing harmless
cross-channel frequency reuse. Checkpoints land in
`all-training/freq-fix-d6-v3/`.

## Configuration queued in `config.py`

| knob | value |
|---|---|
| `frequency_param_mode` | `"anchored"` |
| `freq_anchor_radius_frac` | `0.48` (4% guard gap between adjacent regions) |
| `sine_param_mode` | `"independent"` |
| `d_sine` / `n_samples` | `6` / `2048` |
| `freq_sep_lambda` | `0.05` |
| `clip_per_channel_lambda` | `0.1` |
| `clip_embedding_type` | `"time"` |
| `clip_recon_lambda` | `0.0` |
| `clip_batch_size` / cache chunk | `512` / `512` (direct mode) |
| `clip_lr` / max steps | `1.5e-4` / `50,000` |
| checkpoint base | `all-training/freq-fix-d6-v3/` |

All dataset, optimizer, runtime, and bucketing settings remain the same as the
current d6 recipe. The summed-channel idea is deliberately not enabled here;
v3 remains the multichannel control for that later ablation.

## Architectural delta

Channel `c` uses the fixed midpoint anchor

```text
width     = (f_max - f_min) / d_sine
anchor[c] = f_min + (c + 0.5) * width
f[l,c]    = anchor[c] + 0.48 * width * softsign(raw_f[l,c] + position_bias[l])
```

The position bias is length-aware exactly as before, but is now repeated
inside every channel region instead of flattening all `L*d_sine` waves across
one global band. Separation is averaged over the `L` real token waves within
each channel; padding and cross-channel pairs are excluded.

## Launch

```bash
.venv/bin/python train_clip.py --device cuda
```

Confirm the startup banner points to `freq-fix-d6-v3`, uses the spectral/time
embedding path, and reports the expected BF16/TF32 CUDA runtime.

Validation now reports anchored metrics in place of the obsolete global
sigmoid `sat`/`mid` readout:

- `off-sat`: local softsign inputs with `|raw_f + bias| > 4`.
- `boundary`: waves using more than 90% of their allowed local radius.
- `nn`: mean within-channel nearest-neighbor spacing in Hz.
- `band`: observed frequency span as a fraction of `[f_min, f_max]`.

Run the usual retrieval/STS/compositionality evaluation at the best and final
checkpoints, plus:

```bash
python diagnose_freqs.py all-training/freq-fix-d6-v3/<run_dir>/step_50000.pt
```
