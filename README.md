# Permutohedral Lattice - PyTorch version with CUDA

This is an **UNOFFICIAL GPU** implementation of the permutohedral lattice. The code is based on the original CUDA reference released by the authors, with updates for modern PyTorch/CUDA and a full splat + blur + slice Gaussian/bilateral filter path. For more information, please refer to the project page:
[Fast High-Dimensional Filtering Using the Permutohedral Lattice - Eurographics 2010 (stanford.edu)](https://graphics.stanford.edu/papers/permutohedral/).

## Highlights

- CUDA extension for `float32` tensors shaped `[H, W, pd]` and `[H, W, vd]`.
- Runtime-dispatched `pd` in `[1, 16]` and `vd` in `[1, 8]`.
- Full permutohedral blur pass, not the approximate/no-blur path.
- Backward computes gradients with respect to input values.
- PyTorch autograd wrapper in `plattice.py`.

## Recent CUDA Improvements

This version replaces the older reference-style CUDA path with an optimized full
filter implementation. The main changes are:

- **Full blur behavior**: the forward pass now performs the actual
  permutohedral splat, blur, and slice sequence.
- **Runtime dimensions**: feature dimension `pd` and value dimension `vd` are
  read from input tensors instead of being hard-coded.
- **Backward support**: the extension exposes `backward(feature, grad, weight)`
  and the Python wrapper propagates gradients with respect to input values.
- **PyTorch workspace allocation**: temporary CUDA buffers are allocated as
  PyTorch tensors, allowing PyTorch's caching allocator to avoid repeated
  `cudaMalloc`/`cudaFree` overhead.
- **Precomputed blur neighbors**: blur neighbor indices are resolved once after
  splat and reused by each blur pass, reducing repeated hash-table lookups.
- **Register slice accumulators**: slice kernels use per-thread register
  accumulators instead of shared-memory temporary arrays.

An attempted warp-level splat deduplication variant using `__match_any_sync`
was tested but was slower on the benchmark workload, so the current code keeps
the faster block-level splat dedup path.

## Benchmark

On the tested A5000 setup, the optimized full forward+backward benchmark on the
medium case produced:

```text
median_ms: 1.1410239934921265
score: 876.405759829362
round_total_mad_ms: 0.008383989334106445
forward_median_ms: 0.590175986289978
backward_median_ms: 0.5819200277328491
```

Correctness was checked against calibrated forward and backward outputs on
small, medium, large, and pathological cases. The maximum observed absolute
error was below `5e-7`, with an acceptance tolerance of `5e-3`.

If you find the code useful, please cite

```
@inproceedings{adams2010fast,
  title={Fast high-dimensional filtering using the permutohedral lattice},
  author={Adams, Andrew and Baek, Jongmin and Davis, Myers Abraham},
  booktitle={Computer graphics forum},
  volume={29},
  number={2},
  pages={753--762},
  year={2010},
  organization={Wiley Online Library}
}
```

### Requirements

- PyTorch >= 1.8.0 (not tested on lower versions)
- CUDA-capable GPU
- The provided `setup.py` includes `sm_86` flags for NVIDIA A5000/Ampere.

### Installation

```bash
python setup.py install
```

### Usage

A `torch.autograd.Function` is implemented in `plattice.py`.

```python
import torch
from plattice import PermutoLattice

feature = torch.randn(256, 256, 5, device="cuda", dtype=torch.float32)
values = torch.randn(256, 256, 3, device="cuda", dtype=torch.float32, requires_grad=True)

p = PermutoLattice()
out = p(feature, values)
loss = out.sum()
loss.backward()
```

The input tensors must be contiguous CUDA `float32` tensors:

- `feature`: `[H, W, pd]`, with `1 <= pd <= 16`
- `values`: `[H, W, vd]`, with `1 <= vd <= 8`
