# SOTtools

```{warning}
SOTtools is under active development — the API may change upon updating until we reach v1.0.0.
```

SQUID-based microscopy (SOT, scanning SQUID) is a powerful tool for locally imaging current distributions by mapping their magnetic fields. Interpreting these maps, however, is difficult: the SQUID singles out one particular component of the magnetic field, and inverting these maps back to a current distribution is a non-trivial problem.

**SOTtools** is a Python toolkit for modeling and inverting SOT measurements. It provides:

- FEM mesh generation and current distribution simulation on arbitrarily shaped samples
- Forward models to compute SQUID signal maps from current distributions
- Inverse solvers to reconstruct current distributions from SQUID maps
- Coinversion routines to extract microscope parameters such as height and orientation together with the current distribution

The unique feature of SOTtools is that it is **sample-aware**: it natively accounts for 1D and 2D sample boundaries and confines all currents to the physical sample geometry.

These inversions are made possible by implementing all forward models in [PyTorch](https://pytorch.org/), enabling automatic differentiation. The user-facing API, however, is fully [NumPy](https://numpy.org/)-based, following the conventions of scientific Python.

This documentation comes with extensive tutorial notebooks that demonstrate how to use the module for forward and backward modeling. Check them out below!

Originally, SOTtools was developed for and used in [Rog, Blom *et al.*, arXiv:2606.20157](https://arxiv.org/abs/2606.20157).

## Getting started

```{toctree}
:maxdepth: 2

installation
tutorials/Tutorial_1DForward
tutorials/Tutorial_1DBackward
tutorials/Tutorial_1DCoinversion
```

## Two-dimensional calculations

```{toctree}
:maxdepth: 2
tutorials/Tutorial_2DForward
tutorials/Tutorial_2DBackward_Hydro
tutorials/Tutorial_2DBackward_InhomogeneousResistivity
tutorials/Tutorial_2DBackward_Meissner
tutorials/Tutorial_2DCoinversion_InhomogeneousResistivity
```

## Some notes on kernel computation

```{toctree}
:maxdepth: 2
kernelmath
kernelmath_2d
```

## API 

```{toctree}
:maxdepth: 2
api/index
```

## Citing SOTtools
We are working on a preprint describing SOTtools. In the meantime, please cite the Github repository:
```bibtex
@misc{SOTtools,
  author = {Matthijs Rog},
  title = {SOTtools},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/MatthijsRog/sottools/}
}
```
Alternatively, you can cite [the original paper that sottools was built for](https://arxiv.org/abs/2606.20157).
