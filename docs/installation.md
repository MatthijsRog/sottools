# Installation

## From PyPI

```bash
pip install sottools
```

SOTtools requires Python 3.12 or later.

## GPU acceleration

SOTtools uses PyTorch for its forward models and inverse solvers. By default, `pip install sottools` installs a CPU-only version of PyTorch. For GPU acceleration, install the appropriate CUDA-enabled PyTorch **before** installing SOTtools:

```bash
# Example for CUDA 12.1 — see https://pytorch.org/get-started/locally/ for your setup
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install sottools
```

## Optional dependencies

For interactive use in Jupyter notebooks:

```bash
pip install sottools[jupyter]
```

## Development install

To contribute or run the test suite:

```bash
git clone https://github.com/MatthijsRog/sottools.git
cd sottools
pip install -e .[dev]
```

This installs SOTtools in editable mode along with the development tools (pytest, ruff, mypy).

Run the checks:

```bash
ruff check .
ruff format --check .
mypy src/sottools/ --exclude src/sottools/utils/
pytest tests/ -x --cov=sottools --cov-config=pyproject.toml
```