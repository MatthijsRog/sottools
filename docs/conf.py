project = "sottools"
author = "Matthijs Rog"
release = "0.1.1"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

templates_path = []
exclude_patterns = ["_build"]

html_theme = "furo"

# myst-nb: don't re-execute notebooks, use stored outputs
nb_execution_mode = "off"

# MyST extensions
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Napoleon: NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Intersphinx: link to external docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}