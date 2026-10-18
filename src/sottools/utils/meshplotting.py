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
    n_grid: int = 128,
    density: float = 2.5,
    aspect: str = "equal",
    **kwargs,
) -> None:
    """Plot a streamplot of the current density on the mesh.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    The streamlines are computed from the exact piecewise-constant FEM current:
    every grid point takes the value of the mesh triangle it lies in. No
    smoothing is applied and no values leak across walls, so weak features such
    as vortices are rendered faithfully.

    Parameters
    ----------
    mesh: mesh.SingleBodyMesh
        The mesh on which the current density is defined.
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.
    Jx: np.ndarray
        x-component of the current density, defined on the cell centers.
        Shape (N_t,).
    Jy: np.ndarray
        y-component of the current density, defined on the cell centers.
        Shape (N_t,).
    n_grid: int
        Number of grid points in each direction on which the current is
        sampled for the streamplot. Defaults to 128.
    density: float
        Streamline density passed to ax.streamplot. Higher values seed more
        streamlines, which is needed to resolve small or weak vortices.
        Defaults to 2.5.
    aspect: str
        Aspect ratio of the plot. Defaults to "equal".
    **kwargs:
        Additional keyword arguments passed to ax.tripcolor.
    """
    # Colour map of |J|, plotted with x pointing down and y pointing right
    J_abs = np.sqrt(Jx**2 + Jy**2)
    triangulation_plot = mtri.Triangulation(
        mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.triangles
    )
    tpc = ax.tripcolor(triangulation_plot, J_abs, shading="flat", **kwargs)
    ax.set_aspect(aspect)
    plt.colorbar(tpc, ax=ax)

    # Uniform grid over the mesh, in the mesh's own (x, y) coordinates
    vx, vy = mesh.vertices[:, 0], mesh.vertices[:, 1]
    xi = np.linspace(vx.min(), vx.max(), n_grid)
    yi = np.linspace(vy.min(), vy.max(), n_grid)
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")  # shape (n_grid, n_grid)

    # Find the triangle containing each grid point (-1 outside the mesh)
    triangulation_mesh = mtri.Triangulation(vx, vy, mesh.triangles)
    triangle_index = triangulation_mesh.get_trifinder()(Xi, Yi)
    outside = triangle_index < 0

    # Sample the exact FEM current; mask grid points outside the mesh
    Jx_i = np.ma.masked_where(outside, Jx[triangle_index])
    Jy_i = np.ma.masked_where(outside, Jy[triangle_index])

    # Horizontal axis is y, vertical axis is x (SOT convention)
    ax.streamplot(yi, xi, Jy_i, Jx_i, color="white", density=density, linewidth=0.5)

    ax.invert_yaxis()

def _streamplot(
    mesh: mesh.SingleBodyMesh,
    ax: plt.Axes,
    Jx: np.ndarray,
    Jy: np.ndarray,
    n_grid=128,
    aspect: str = "equal",
    **kwargs,
) -> None:
    """Plot a streamplot of the current density on the mesh.

    WARNING: DEPRECATED AS OF VERSION 0.1.5. Use the new streamplot().

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
