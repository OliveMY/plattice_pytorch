# Permutohedral Lattice — PyTorch version with CUDA

This is an **UNOFFICIAL GPU** implementation of the permutohedral lattice. The code is mainly a wrapper of original code released by the authors, with modifications to out-dated cuda function. For more information, please refer to the project page :
[Fast High-Dimensional Filtering Using the Permutohedral Lattice - Eurographics 2010 (stanford.edu)](https://graphics.stanford.edu/papers/permutohedral/).


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

PyTorch ≥ 1.8.0 (Not sure if it works on lower versions)

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

### License
