import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.interpolate
from matplotlib.path import Path

import sottools.mesh as mesh


def plot_mesh(
    mesh: mesh.SingleBodyMesh, ax: plt.Axes, aspect: str = "equal", **kwargs
) -> None:
    """Plot an outline of the mesh on the given axes.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.

    aspect: str
        Aspect ratio of the plot. Defaults to "equal".

    **kwargs:
        Additional keyword arguments passed to ax.triplot.
    """
    triangulation = mtri.Triangulation(
        mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.triangles
    )

    kwargs.setdefault("linestyle", "-")
    ax.triplot(triangulation, **kwargs)

    ax.set_aspect(aspect)

    ax.invert_yaxis()


def plot_field(
    mesh: mesh.SingleBodyMesh,
    ax: plt.Axes,
    field: np.ndarray,
    aspect: str = "equal",
    **kwargs,
) -> None:
    """Plot a vertex-centered field on the mesh.

    Uses tripcolor, thereby visualizing the assumed piecewise linear
    structure of a field in FEM implementations.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.

    field: np.ndarray
        Field defined on the vertices of the mesh. Shape (N_v,).

    aspect: str
        Aspect ratio of the plot. Defaults to "equal".

    **kwargs:
        Additional keyword arguments passed to ax.tripcolor.
    """
    triangulation = mtri.Triangulation(
        mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.triangles
    )
    tpc = ax.tripcolor(triangulation, field, shading="gouraud", **kwargs)
    ax.set_aspect(aspect)
    ax.invert_yaxis()
    plt.colorbar(tpc, ax=ax)


def streamplot(
    mesh: mesh.SingleBodyMesh,
    ax: plt.Axes,
    Jx: np.ndarray,
    Jy: np.ndarray,
    n_grid=128,
    aspect: str = "equal",
    **kwargs,
) -> None:
    """Plot a streamplot of the current density on the mesh.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.

    Jx: np.ndarray
        x-component of the current density, defined on the cell centers.
        Shape (N_t,).
    Jy: np.ndarray
        y-component of the current density, defined on the cell centers.
        Shape (N_t,).
    n_grid: int
        Number of grid points in each direction for the streamplot.
        Defaults to 128.
    aspect: str
        Aspect ratio of the plot. Defaults to "equal".
    **kwargs:
        Additional keyword arguments passed to ax.tripcolor.
    """
    J_abs = np.sqrt(Jx**2 + Jy**2)
    triangulation = mtri.Triangulation(
        mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.triangles
    )

    tpc = ax.tripcolor(triangulation, J_abs, shading="flat", **kwargs)
    ax.set_aspect(aspect)
    plt.colorbar(tpc, ax=ax)

    # For streamplot, we must first interpolate currents on a uniform grid
    cx, cy = mesh.centers[:, 0], mesh.centers[:, 1]
    xmin, xmax = cx.min(), cx.max()
    ymin, ymax = cy.min(), cy.max()
    xi = np.linspace(xmin, xmax, n_grid)
    yi = np.linspace(ymin, ymax, n_grid)
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")
    Jx_i = scipy.interpolate.griddata((cx, cy), Jx, (Xi, Yi), method="cubic")
    Jy_i = scipy.interpolate.griddata((cx, cy), Jy, (Xi, Yi), method="cubic")

    boundary_path = Path(mesh.boundary_vertices)
    mask = ~boundary_path.contains_points(
        np.column_stack([Xi.ravel(), Yi.ravel()])
    ).reshape(Xi.shape)
    Jx_i[mask] = np.nan
    Jy_i[mask] = np.nan
    ax.streamplot(yi, xi, Jy_i, Jx_i, color="white", density=0.65, linewidth=0.5)

    ax.invert_yaxis()
