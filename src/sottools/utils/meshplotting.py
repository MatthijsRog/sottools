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


def plot_tc_field(
    mesh: mesh.SingleBodyMesh,
    ax: plt.Axes,
    field: np.ndarray,
    aspect: str = "equal",
    **kwargs,
) -> None:
    """Plot a triangle-centered field on the mesh.

    Uses tripcolor, thereby visualizing the assumed piecewise linear
    structure of a field in FEM implementations.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    Parameters
    ----------
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.

    field: np.ndarray
        Field defined on the triangles of the mesh. Shape (N_t,).

    aspect: str
        Aspect ratio of the plot. Defaults to "equal".

    **kwargs:
        Additional keyword arguments passed to ax.tripcolor.
    """
    triangulation = mtri.Triangulation(
        mesh.vertices[:, 1], mesh.vertices[:, 0], mesh.triangles
    )
    tpc = ax.tripcolor(triangulation, field, shading="flat", **kwargs)
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


def quiverplot(
    mesh: mesh.SingleBodyMesh,
    ax: plt.Axes,
    g: np.ndarray,
    n_grid: int = 512,
    arrow_spacing: float | None = None,
    scale: float | None = None,
    kind: str = "min_E",
    aspect: str = "equal",
    quiver_kwargs: dict | None = None,
    **kwargs,
) -> None:
    """Plot a smoothed current density, derived from the streamfunction.

    Uses SOT coordinate conventions: right handed coordinate system
    with x pointing down and y pointing right.

    The vertex-centered streamfunction is interpolated with a C1 piecewise
    cubic (Clough-Tocher) interpolant. The current density J = curl(g z_hat),
    i.e. Jx = dg/dy and Jy = -dg/dx, is obtained from the analytic gradient of
    this interpolant. This yields a continuous current, in contrast to the
    piecewise-constant current of the linear FEM representation. The colour
    map shows |J| on a fine grid; arrows show J on a coarser regular array.

    Parameters
    ----------
    mesh: mesh.SingleBodyMesh
        The mesh on which the streamfunction is defined.
    ax: plt.Axes
        Matplotlib axes to plot on. Mutable.
    g: np.ndarray
        Streamfunction defined on the vertices of the mesh. Shape (N_v,).
    n_grid: int
        Number of grid points in each direction on which |J| is evaluated
        for the colour map. Defaults to 512.
    arrow_spacing: float | None
        Spacing of the arrow array, in mesh length units. Defaults to 1/20
        of the largest mesh extent.
    scale: float | None
        Current density per unit of plot length. Arrow length is |J| / scale
        in mesh length units, so larger values give shorter arrows. None
        uses matplotlib's autoscaling.
    kind: str
        Smoothing algorithm of mtri.CubicTriInterpolator: "min_E" (smoothest,
        solves a global sparse system) or "geom" (faster, local).
        Defaults to "min_E".
    aspect: str
        Aspect ratio of the plot. Defaults to "equal".
    quiver_kwargs: dict | None
        Additional keyword arguments passed to ax.quiver.
    **kwargs:
        Additional keyword arguments passed to ax.pcolormesh.
    """
    # C1 cubic interpolant of g, in the mesh's own (x, y) coordinates
    vx, vy = mesh.vertices[:, 0], mesh.vertices[:, 1]
    triangulation_mesh = mtri.Triangulation(vx, vy, mesh.triangles)
    interpolator = mtri.CubicTriInterpolator(triangulation_mesh, g, kind=kind)

    # Fine uniform grid over the mesh; gradient is masked outside the mesh
    xi = np.linspace(vx.min(), vx.max(), n_grid)
    yi = np.linspace(vy.min(), vy.max(), n_grid)
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")  # shape (n_grid, n_grid)
    dg_dx, dg_dy = interpolator.gradient(Xi, Yi)
    J_abs_i = np.ma.sqrt(dg_dx**2 + dg_dy**2)

    # Colour map of |J|; horizontal axis is y, vertical axis is x (SOT convention)
    pcm = ax.pcolormesh(yi, xi, J_abs_i, shading="auto", **kwargs)
    ax.set_aspect(aspect)
    plt.colorbar(pcm, ax=ax)

    # Regular arrow array, centred within the mesh extent
    if arrow_spacing is None:
        arrow_spacing = max(np.ptp(vx), np.ptp(vy)) / 20
    xq = np.arange(vx.min() + arrow_spacing / 2, vx.max(), arrow_spacing)
    yq = np.arange(vy.min() + arrow_spacing / 2, vy.max(), arrow_spacing)
    Xq, Yq = np.meshgrid(xq, yq, indexing="ij")

    # Evaluate the current exactly at the arrow points (not subsampled from grid)
    dg_dx_q, dg_dy_q = interpolator.gradient(Xq, Yq)
    Jx_q = dg_dy_q
    Jy_q = -dg_dx_q

    # angles="xy" makes arrows follow data coordinates, so they stay correct
    # after the vertical axis is inverted
    quiver_kwargs = {} if quiver_kwargs is None else dict(quiver_kwargs)
    quiver_kwargs.setdefault("color", "white")
    quiver_kwargs.setdefault("pivot", "middle")
    ax.quiver(
        Yq,
        Xq,
        Jy_q,
        Jx_q,
        angles="xy",
        scale_units="xy",
        scale=scale,
        **quiver_kwargs,
    )

    ax.invert_yaxis()
