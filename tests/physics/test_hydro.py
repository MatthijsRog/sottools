import numpy as np
import pytest

import sottools.mesh as Mesh


def test_v_matrix(bar_mesh):
    rng = np.random.default_rng(0)

    ten = rng.uniform(0.1, 3.0, size=1)
    D = rng.uniform(0.1, 3.0, size=1)
    g_left = rng.uniform(-5, 5, size=1)
    g_right = rng.uniform(-5, 5, size=1)

    solver = Mesh.HydroSolver(bar_mesh, ten, D)

    g = np.zeros(bar_mesh.vertices.shape[0])
    g[bar_mesh.dirichletleft_mask] = g_left
    g[bar_mesh.dirichletright_mask] = g_right
    g[bar_mesh.free_mask] = rng.uniform(-5, 5, size=np.sum(bar_mesh.free_mask))
    V = solver._v_matrix(g)

    # Verify that V * identity = 0
    N_v = bar_mesh.vertices.shape[0]
    assert np.allclose(V @ np.ones(N_v), 0, atol=1e-6)

    # Verify that the interior block of V is skew-symmetric for random g
    interior_idx = np.setdiff1d(np.arange(N_v), np.unique(bar_mesh.boundary_segments))
    row_sel = np.isin(
        bar_mesh.free_idx, interior_idx
    )  # rows of V are ordered by free_idx
    V_int = V[row_sel][:, interior_idx]
    assert abs(V_int + V_int.T).max() < 1e-6

    # Now make g linear in x
    g = bar_mesh.vertices[:, 1]  # linear in y
    omega = bar_mesh.vertices[:, 0]  # linear in x

    # Verify that for this g, V omega = M * 1, where M is the mass matrix
    V = solver._v_matrix(g)
    assert np.allclose(
        V @ omega, bar_mesh.mass_matrix[bar_mesh.free_idx] @ np.ones(N_v)
    )


@pytest.mark.parametrize("D", [1, 2, 3, 4])
def test_hydro_solver(bar_mesh, D):
    ten = 0.0
    solver = Mesh.HydroSolver(bar_mesh, ten, D)
    res = solver.solve(0.0, 1.0)
    assert res["converged"]

    # Expected streamfunction: integral of the Gurzhi profile across the bar
    xv, yv = bar_mesh.vertices[:, 0], bar_mesh.vertices[:, 1]
    mask = (xv > 3.0) & (xv < 5.0)
    y = yv[mask]
    g = res["streamfunction"][mask]

    W = 3.0  # bar width, from the fixture geometry
    y_mid = W / 2  # centreline
    total_current = 1.0  # total current = right_bc - left_bc
    J0 = total_current / (W - 2 * D * np.tanh(W / (2 * D)))
    g_expected = J0 * (
        y - D * (np.sinh((y - y_mid) / D) + np.sinh(y_mid / D)) / np.cosh(W / (2 * D))
    )

    assert np.allclose(g, g_expected, rtol=0, atol=0.01 * total_current)


def test_no_advection_straight_channel():
    """Advective term vanishes for unidirectional flow.

    Poiseuille/Gurzhi flow in a straight channel is an exact solution for any
    Re, because (J . nabla) J = 0 for unidirectional flow. On an unstructured
    mesh this holds only up to discretization error, so we check that the
    deviation between ten > 0 and ten = 0 shrinks under mesh refinement.
    """
    BM = Mesh.SimplyConnectedCurrentMesh.BoundaryMarker
    vertices = [[0, 0], [0, 3], [8, 3], [8, 0]]
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
    ]

    D, ten = 1.0, 1.0
    left_bc, right_bc = 0.0, 1.0

    deviations = []
    for max_area in [0.1, 0.025]:
        mesh = Mesh.SimplyConnectedCurrentMesh(
            vertices, segments, segment_markers, max_area=max_area
        )
        res_stokes = Mesh.HydroSolver(mesh, 0.0, D).solve(
            left_bc, right_bc, rtol=1e-10, max_iter=200
        )
        res_adv = Mesh.HydroSolver(mesh, ten, D).solve(
            left_bc, right_bc, rtol=1e-10, max_iter=200
        )
        assert res_stokes["converged"] and res_adv["converged"]
        deviations.append(
            np.max(np.abs(res_adv["streamfunction"] - res_stokes["streamfunction"]))
        )

    # Halving h should cut the deviation by ~4x (observed ~4.7x); demand at least 3x
    assert deviations[1] < deviations[0] / 3


def test_advection_constriction():
    """Advection makes a difference for flow through a constriction."""
    BM = Mesh.SimplyConnectedCurrentMesh.BoundaryMarker
    vertices = [
        [0, 0],
        [0, 2],
        [3, 2],
        [3.5, 1.2],
        [4.5, 1.2],
        [5.0, 2],
        [8, 2],
        [8, 0],
        [5.0, 0],
        [4.5, 0.8],
        [3.5, 0.8],
        [3, 0],
    ]
    segments = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [11, 0],
    ]
    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.DIRICHLET_RIGHT,
        BM.DIRICHLET_RIGHT,
        BM.DIRICHLET_RIGHT,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
    ]

    D, ten = 1.0, 1.0
    mesh = Mesh.SimplyConnectedCurrentMesh(
        vertices, segments, segment_markers, max_area=0.025
    )

    def asymmetry(res):
        # int |J|^2 dA downstream of the constriction centre (x = 4) minus upstream
        J2_dA = (res["Kx"] ** 2 + res["Ky"] ** 2) * mesh.areas
        x = mesh.centers[:, 0]
        return np.sum(J2_dA[x > 4.0]) - np.sum(J2_dA[x < 4.0])

    def solve(ten, left_bc, right_bc):
        res = Mesh.HydroSolver(mesh, ten, D).solve(
            left_bc, right_bc, rtol=1e-10, max_iter=300
        )
        assert res["converged"]
        return res

    res_stokes = solve(0.0, 0.0, 1.0)
    A_stokes = asymmetry(res_stokes)  # mesh-asymmetry baseline
    scale = np.sum((res_stokes["Kx"] ** 2 + res_stokes["Ky"] ** 2) * mesh.areas)

    dA_forward = asymmetry(solve(ten, 0.0, 1.0)) - A_stokes
    dA_reverse = asymmetry(solve(ten, 1.0, 0.0)) - A_stokes

    # Advection must break fore-aft symmetry: observed |dA| ~ 0.031 ~ 4e-3 * scale
    assert abs(dA_forward) > 1e-3 * scale
    # ...and reversing the current must mirror it
    assert np.isclose(dA_forward, -dA_reverse, rtol=0.1)
