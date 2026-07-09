import matplotlib.tri as mtri
import numpy as np
import pytest

import sottools.mesh as Mesh


def test_bar_current(bar_mesh_london):
    rng = np.random.default_rng(0)
    (Curr,) = rng.uniform(-5, 5, size=1)
    pearllength = rng.uniform(0.1, 3.0, size=1)

    # First verify that in this mesh, current indeed flows homogeneously
    # in +x

    mesh_width = np.max(bar_mesh_london.vertices[:, 1]) - np.min(
        bar_mesh_london.vertices[:, 1]
    )

    solver = Mesh.LondonSolver(bar_mesh_london, pearllength)
    Bz = np.zeros(solver.mesh.vertices.shape[0])
    g = solver._solve_streamfunction(
        0.0, Curr, Bz=Bz
    )  # Find streamfunction for given total current and zero self-field

    # Verify g is indeed Curr * (y-ymin) / mesh_width
    ymin = np.min(bar_mesh_london.vertices[:, 1])
    g_expected = Curr * (bar_mesh_london.vertices[:, 1] - ymin) / mesh_width

    assert np.allclose(g, g_expected, atol=1e-6)


def test_bar_current_field(bar_mesh_london):
    """Test against analytical result for infinitely long 2D strip.

    In the infinitely long 2D strip, the OOP magnetic field is given by:

    2mu0*K/4pi * {ln(y-y0)-ln(y1-y)}

    where y0 and y1 are the left and right edges of the strip, respectively.
    """

    rng = np.random.default_rng(0)
    (Curr,) = rng.uniform(-5, 5, size=1)
    pearllength = rng.uniform(0.1, 3.0, size=1)
    mesh_width = np.max(bar_mesh_london.vertices[:, 1]) - np.min(
        bar_mesh_london.vertices[:, 1]
    )
    y0 = np.min(bar_mesh_london.vertices[:, 1])
    y1 = np.max(bar_mesh_london.vertices[:, 1])
    x0 = np.min(bar_mesh_london.vertices[:, 0])
    x1 = np.max(bar_mesh_london.vertices[:, 0])
    center_x = (x0 + x1) / 2

    K_expected = Curr / mesh_width * np.ones(len(bar_mesh_london.triangles))
    mu0 = 4 * np.pi * 1e-1  # Standard units um, uA, uT

    solver = Mesh.LondonSolver(bar_mesh_london, pearllength)
    solver.MU0 = mu0
    g = solver._solve_streamfunction(
        0.0, Curr, Bz=np.zeros(solver.mesh.vertices.shape[0])
    )
    Kx, Ky = solver.mesh.field_curl(g)
    Bz = solver._calculate_Bz(Kx, Ky)

    # Compute expected Bz for a line from y0 to y1 along center_x
    N_points = 100
    triangulation = mtri.Triangulation(
        bar_mesh_london.vertices[:, 0],
        bar_mesh_london.vertices[:, 1],
        bar_mesh_london.triangles,
    )
    fz = mtri.LinearTriInterpolator(triangulation, Bz)
    yax = np.linspace(y0, y1, N_points)
    Bz_simulated = fz(center_x * np.ones(N_points), yax)

    eps = 1e-12  # Small epsilon to avoid log(0)
    Bz_expected = (
        mu0
        * K_expected[0]
        / (2 * np.pi)
        * (np.log(yax - y0 + eps) - np.log(y1 - yax + eps))
    )

    atol = 0.1 * np.max(Bz_expected)

    # We check if every point is within atol of expectation
    # With exception of first and last point, where the expectation diverges
    assert np.allclose(Bz_simulated[1:-1], Bz_expected[1:-1], atol=atol)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bar_current_profile(bar_mesh_london, seed):
    """Test that the current profile in a full London simulation follows the
    Rhoderick-Wilson profile for a few trial pearl lengths."""

    rng = np.random.default_rng(seed)
    (Curr,) = rng.uniform(-5, 5, size=1)
    pearllength = rng.uniform(0.075, 0.2, size=1)
    mesh_width = np.max(bar_mesh_london.vertices[:, 1]) - np.min(
        bar_mesh_london.vertices[:, 1]
    )
    y0 = np.min(bar_mesh_london.vertices[:, 1])
    y1 = np.max(bar_mesh_london.vertices[:, 1])
    x0 = np.min(bar_mesh_london.vertices[:, 0])
    x1 = np.max(bar_mesh_london.vertices[:, 0])
    center_x = (x0 + x1) / 2

    solver = Mesh.LondonSolver(bar_mesh_london, pearllength)
    mu0 = 4 * np.pi * 1e-1  # Standard units um, uA, uT
    solver.MU0 = mu0
    res = solver.solve(0.0, Curr)

    assert res["converged"] is True

    # Extract current density along center line of bar for a few points
    # Around the center and average
    Nlines = 21
    teststripheight = 0.3 * (x1 - x0)
    xlines = np.linspace(center_x + teststripheight, center_x - teststripheight, Nlines)
    N_points = 100
    triangulation = mtri.Triangulation(
        bar_mesh_london.vertices[:, 0],
        bar_mesh_london.vertices[:, 1],
        bar_mesh_london.triangles,
    )
    trifinder = triangulation.get_trifinder()
    yax = np.linspace(y0, y1, N_points)
    currentprofile = np.zeros(N_points)
    for xline in xlines:
        tri_indices = trifinder(xline * np.ones(N_points), yax)
        currentprofile += res["Kx"][tri_indices]
    currentprofile /= Nlines

    # Rhoderick-Wilson approximation
    w = mesh_width
    L = pearllength
    y_center = (y0 + y1) / 2
    y_rel = yax - y_center  # centered coordinate

    # J(0) from total current normalization
    denom = w * np.arcsin(1 - 2 * L / w) + 2 * (np.exp(0.5) - 1) * np.sqrt(
        L * w
    ) / np.sqrt(1 - L / w)
    J0 = Curr / denom

    # J(w/2) matched value
    Jedge = J0 * np.exp(0.5) * 0.5 / np.sqrt(L / w - (L / w) ** 2)

    # Build piecewise profile
    rw_profile = np.zeros_like(yax)
    abs_y = np.abs(y_rel)
    bulk = abs_y < (w / 2 - L)
    edge = ~bulk

    rw_profile[bulk] = J0 / np.sqrt(1 - (2 * y_rel[bulk] / w) ** 2)
    rw_profile[edge] = Jedge * np.exp(-(w / 2 - abs_y[edge]) / (2 * L))
    compare_mask = (
        abs_y < 0.98 * w / 2
    )  # Last cell always underestimates and should not be tested against

    assert np.allclose(
        currentprofile[compare_mask], rw_profile[compare_mask], rtol=0.15
    )


def test_alpha_invaraince(bar_mesh_london):
    # Different values of the damping parameter alpha should give the same results
    rng = np.random.default_rng(0)
    (Curr,) = rng.uniform(-5, 5, size=1)
    pearllength = rng.uniform(0.075, 1.0, size=1)

    solver = Mesh.LondonSolver(bar_mesh_london, pearllength)
    mu0 = 4 * np.pi * 1e-1  # Standard units um, uA, uT
    solver.MU0 = mu0

    results = []
    for alpha in [0.1, 0.2, 0.3]:
        res = solver.solve(0.0, Curr, alpha=alpha, max_iter=400)
        assert res["converged"] is True, f"Solver did not converge for alpha={alpha}"
        results.append(res)

    assert np.allclose(results[0]["Kx"], results[1]["Kx"], rtol=1e-4)
    assert np.allclose(results[0]["Kx"], results[2]["Kx"], rtol=1e-4)
    assert np.allclose(results[0]["Ky"], results[1]["Ky"], rtol=1e-4, atol=1e-5)
    assert np.allclose(results[0]["Ky"], results[2]["Ky"], rtol=1e-4, atol=1e-5)
    assert np.allclose(results[0]["Bz"], results[1]["Bz"], rtol=1e-4, atol=1e-5)
    assert np.allclose(results[0]["Bz"], results[2]["Bz"], rtol=1e-4, atol=1e-5)
    assert np.allclose(
        results[0]["streamfunction"], results[1]["streamfunction"], rtol=1e-4
    )
    assert np.allclose(
        results[0]["streamfunction"], results[2]["streamfunction"], rtol=1e-4
    )
