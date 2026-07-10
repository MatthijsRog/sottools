"""Collection of tools for current reconstruction using SQUID-on-tip.

Mesh.py contains classes to define a sample geometry (as a FEM mesh) and
to solve the London equations on that mesh.

Forward.py contains classes to compute the forward Biot-Savart problem for a
given current distribution in 1D or 2D.

Backward.py contains classes to solve the inverse Biot-Savart problem, i.e. to
reconstruct the current distribution from a measured SOT signal.

The utils folder contains various utility functions for plotting and mesh
generation.
"""

__version__ = "0.1.1"
