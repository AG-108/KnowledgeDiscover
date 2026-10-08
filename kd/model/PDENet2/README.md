# PDE-Net 2.0 runtime

This directory retains the PDE-Net 2.0 model and aTEAM runtime used by
[`PDENetModel`](../kd_pdenet.py). The project adapter provides training and
prediction on supported 2D grid datasets; benchmark configuration lives in
[`pdenet.json`](../../../configs/benchmark/baselines/pdenet.json).

The original equation-specific training, SLURM, comparison and debug scripts
and standalone aTEAM tests have been removed. Project regression tests live in
[`tests/`](../../../tests/). The generators in `aTEAM/pdetools/example/` remain
runtime dependencies of `setenv.py`.

The upstream implementation was developed with Python 3.7 and PyTorch 1.1.0.
This vendored runtime includes compatibility changes for the project's Python
3.10 environment; use the project environment instructions in
[`docs/benchmark.md`](../../../docs/benchmark.md).

Upstream sources: [PDE-Net](https://github.com/ZichaoLong/PDE-Net),
[aTEAM](https://github.com/ZichaoLong/aTEAM).
The aTEAM license is retained in [`aTEAM/LICENSE`](aTEAM/LICENSE).

Please cite the original method when using it:

```bibtex
@article{long2019pde,
  title={PDE-Net 2.0: Learning PDEs from data with a numeric-symbolic hybrid deep network},
  author={Long, Zichao and Lu, Yiping and Dong, Bin},
  journal={Journal of Computational Physics},
  pages={108925},
  year={2019},
  publisher={Elsevier}
}
```
