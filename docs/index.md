# SOTtools

```{warning}
SOTtools is under active development — the API may change between minor versions.
```

SQUID-based microscopy (SOT, scanning SQUID) is a powerful tool for locally imaging current distributions by mapping their magnetic fields. Interpreting these maps, however, is difficult: the SQUID singles out one particular component of the magnetic field, and inverting these maps back to a current distribution is a non-trivial problem.

**SOTtools** is a Python toolkit for modeling and inverting SOT measurements. It provides:

- FEM mesh generation and current distribution simulation on arbitrarily shaped samples
- Forward models to compute SQUID signal maps from current distributions
- Inverse solvers to reconstruct current distributions from SQUID maps

The unique feature of SOTtools is that it is **sample-aware**: it natively accounts for 1D and 2D sample boundaries and confines all currents to the physical sample geometry.

These inversions are made possible by implementing all forward models in [PyTorch](https://pytorch.org/), enabling automatic differentiation. The user-facing API, however, is fully [NumPy](https://numpy.org/)-based, following the conventions of scientific Python.

Originally, SOTtools was developed for and used in [Rog, Blom *et al.*, arXiv:2606.20157](https://arxiv.org/abs/2606.20157).

## Getting started

```{toctree}
:maxdepth: 2

installation
tutorials/Tutorial_1DForward
tutorials/Tutorial_1DBackward
tutorials/Tutorial_2DForward
api/index
```