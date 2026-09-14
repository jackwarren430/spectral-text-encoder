# DGX Spark training

The default `Config` is tuned for the local DGX Spark (GB10, ARM64, CUDA 13)
while keeping the `freq-fix-d6` experiment's batch membership, losses, and
model dimensions unchanged.

## Environment

Use a CUDA 13 PyTorch build with native SM 12.1 support. The system Python does
not include PyTorch, so create a project-local environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --pre torch \
  --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install --python .venv/bin/python \
  "transformers>=4.40" "datasets>=2.18" tqdm matplotlib
```

Preflight:

```bash
.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

Expected: CUDA 13.x, `NVIDIA GB10`, and `True` for BF16.

## Launch

```bash
.venv/bin/python train_clip.py --device cuda
```

The startup banner must report all of the following before training:

```text
device=cuda precision=bf16 tf32=True gpu=NVIDIA GB10 sm=12.1
workers=8 pin_memory=True
optimizer=fused AdamW
mode=grad_cache(chunk=64) enc=spectral freq=anchored channels=sum recon_λ=0.5
```

Resume after an interruption with the timestamped run directory:

```bash
.venv/bin/python train_clip.py \
  --resume all-training/freq-fix-d6/<run_dir> --device cuda
```

Old run folders that predate the Spark runtime settings resume in FP32. This is
intentional: silently switching precision halfway through an old run would make
it a different experiment.

## Optimizations enabled

- BF16 autocast for the transformer and contrastive matrix multiplies. The
  waveform parameter head and sine synthesis remain FP32 because BF16 spacing
  near 960 Hz is too coarse for this representation.
- TF32 for remaining FP32 matrix multiplications and fused CUDA AdamW.
- Eight persistent loader workers, four-batch prefetching, page-locked batches,
  and non-blocking CUDA transfers.
- Batched per-channel InfoNCE and upper-triangle frequency pairs, eliminating
  redundant kernel launches and duplicate pair calculations.
- In-batch execution bucketing at 16-token widths. It trims padding for the
  encoder and synthesis, then restores original order before InfoNCE. It does
  **not** length-sort the dataset or alter which 512 negatives share a batch.
- Reconstruction uses signal-level GradCache with 64-example chunks. The
  contrastive objective still sees all 512 negatives, while decoder graphs are
  retained for only one chunk at a time.

`torch.compile` is not enabled for this run. Sentence widths vary and internal
length buckets already remove the dominant padding cost; compiler recompilation
would add risk to a controlled training run without a demonstrated gain.

## Local acceptance benchmark

On the GB10 with CUDA 13 PyTorch 2.11 and the installed 2.14 nightly, a
synthetic batch matching the observed shape (batch 512, max length 91, mean
length 13.3) measured after warmup:

| path | seconds/step | peak allocated |
|---|---:|---:|
| BF16, no execution bucketing | 1.95 | 32.8 GB |
| BF16, 16-token execution buckets | 0.45–0.51 | 8.1 GB |

This is a kernel-path smoke benchmark, not an end-to-end throughput guarantee;
validation, checkpoints, data loading, and other GPU workloads add wall time.

The summed-signal reconstruction smoke (full 512d/8-layer encoder and decoder,
batch 512, chunk 64, BF16, realistic padded lengths) completed a forward,
backward, gradient clip, and fused-AdamW update in 10.0 seconds with 15.8 GiB
peak allocated. The decoder received finite gradients and its one-channel
input projection changed after the step.
