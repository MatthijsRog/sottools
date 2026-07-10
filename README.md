[![CI](https://github.com/MatthijsRog/sottools/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthijsRog/sottools/actions)
[![PyPI](https://img.shields.io/pypi/v/sottools)](https://pypi.org/project/sottools/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-blue)](http://mypy-lang.org/)

# SOTtools

⚠️ **SOTtools is under development** — API may change between minor versions, and old code may break.

SQUID based microscopy (SOT, scanning SQUID) is a powerful tool for locally imaging current distributions by mapping
their magnetic fields. Interpreting these maps, however, is difficult: the SQUID singles out one particular component of
the magnetic field, and inverting these maps is a difficult task.

SOTtools is a toolkit for modeling and inverting SOT measurements. It contains:

- A set of tools for generating FEM meshes and simulating current distributions on these meshes
- Tools to compute SQUID maps from current distributions
- Tools to invert SQUID maps back to current distributions

The unique feature of SOTtools is that it is sample-aware: it natively takes into account 1D and 2D sample boundaries
and confines all currents to the sample.

These inversions are possible only because the forward models are implemented fully in PyTorch, which allows for
autodifferentiation. The user API, however, is fully in NumPy, the standard in scientific Python.

SOTtools is still under development and contributions are welcome.

## Install

SOTtools is available on PyPI and can be installed with pip:

```bash
pip install sottools
```

If GPU acceleration is desired, install the PyTorch version with CUDA support first, then install SOTtools.

For easier usage with Jupyter notebooks, install the optional dependencies:

```bash
pip install sottools[jupyter]
```

## Documentation

Coming soon at [sottools.readthedocs.io](https://sottools.readthedocs.io)
