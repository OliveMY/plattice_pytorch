# Permutohedral Lattice - PyTorch version with CUDA

This is an **UNOFFICIAL GPU** implementation of the permutohedral lattice. The code is based on the original CUDA reference released by the authors, with updates for modern PyTorch/CUDA and a full splat + blur + slice Gaussian/bilateral filter path. For more information, please refer to the project page:
[Fast High-Dimensional Filtering Using the Permutohedral Lattice - Eurographics 2010 (stanford.edu)](https://graphics.stanford.edu/papers/permutohedral/).

Current implementation notes:

- CUDA extension for `float32` tensors shaped `[H, W, pd]` and `[H, W, vd]`.
- Runtime-dispatched `pd` in `[1, 16]` and `vd` in `[1, 8]`.
- Full permutohedral blur pass, not the approximate/no-blur path.
- Backward computes gradients with respect to input values.
- Optimized workspace allocation via PyTorch CUDA tensors, precomputed blur neighbors, and register slice accumulators.


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

PyTorch >= 1.8.0 (not tested on lower versions)

### Installation

```jsx
python setup.py install
```

### Usage

A torch.autograd.Function is implemented in plattice.py. Please refer to the file.
```
import torch
from plattice import PermutoLattice

p = PermutoLattice()
out = p(feature, values)
```
