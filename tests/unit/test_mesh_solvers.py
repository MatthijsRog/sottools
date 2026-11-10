import numpy as np
import pytest

from sottools.mesh import HydroSolver, InhomogeneousOhmicSolver, LondonSolver


@pytest.mark.parametrize(
    "solver, tol", [("ohmic", 1e-10), ("hydro", 1e-10), ("london", 1e-6)]
)
def test_linear_limit(solver, tol, bar_mesh):
    # In these limits every solver reduces to Laplace; on the 8x3 bar (left wall
    # y=0, right wall y=3) the exact P1 solution is g = I*y/3
    current = 2.5
    boundary_conditions = [0, current]
    if solver == "ohmic":
        rho = np.full(len(bar_mesh.triangles), 3.7)
        res = InhomogeneousOhmicSolver(bar_mesh, rho).solve(
            boundary_conditions=boundary_conditions
        )
    elif solver == "hydro":
        res = HydroSolver(bar_mesh, ten=0.0, D=0.0).solve(
            boundary_conditions=boundary_conditions, alpha=1.0
        )
        assert res["converged"]
    else:
        res = LondonSolver(bar_mesh, Pearllength=1e8).solve(
            boundary_conditions=boundary_conditions
        )
        assert res["converged"]

    y = bar_mesh.vertices[:, 1]
    np.testing.assert_allclose(
        res["streamfunction"], current * y / 3, rtol=0, atol=tol * current
    )
    # g_right > g_left: current flows in +x, from NEUMANN_IN (x=0) to NEUMANN_OUT
    np.testing.assert_allclose(res["Kx"], current / 3, rtol=0, atol=tol * current)
    np.testing.assert_allclose(res["Ky"], 0.0, rtol=0, atol=tol * current)


@pytest.mark.parametrize("solver, tol", [("hydro", 1e-10), ("london", 1e-6)])
def test_laplace_limit_multiterminal(solver, tol, tjunction_mesh):
    # In the Laplace limit, both reduce to Ohm's law with uniform rho
    wall_values = [0.0, 1.0, 0.4]
    rho = np.ones(len(tjunction_mesh.triangles))
    reference = InhomogeneousOhmicSolver(tjunction_mesh, rho).solve(wall_values)
    if solver == "hydro":
        res = HydroSolver(tjunction_mesh, ten=0.0, D=0.0).solve(wall_values, alpha=1.0)
    else:
        res = LondonSolver(tjunction_mesh, Pearllength=1e8).solve(wall_values)
    assert res["converged"]
    np.testing.assert_allclose(
        res["streamfunction"], reference["streamfunction"], rtol=0, atol=tol
    )


def test_ohmic_linearity_and_gauge(tjunction_mesh):
    rho = 1.0 + tjunction_mesh.centers[:, 0] ** 2
    solver = InhomogeneousOhmicSolver(tjunction_mesh, rho)
    a = np.array([0.0, 1.0, 0.4])
    b = np.array([0.3, -0.2, 0.7])
    res_a = solver.solve(a)
    res_b = solver.solve(b)
    res_ab = solver.solve(a + b)
    res_shifted = solver.solve(a + 2.5)

    np.testing.assert_allclose(
        res_ab["streamfunction"],
        res_a["streamfunction"] + res_b["streamfunction"],
        rtol=0,
        atol=1e-10,
    )
    # Adding a constant to all walls shifts g but leaves the current unchanged
    np.testing.assert_allclose(
        res_shifted["streamfunction"], res_a["streamfunction"] + 2.5, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(res_shifted["Kx"], res_a["Kx"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(res_shifted["Ky"], res_a["Ky"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("solver", ["ohmic", "hydro", "london"])
def test_wrong_number_of_wall_values(solver, tjunction_mesh):
    if solver == "ohmic":
        obj = InhomogeneousOhmicSolver(
            tjunction_mesh, np.ones(len(tjunction_mesh.triangles))
        )
    elif solver == "hydro":
        obj = HydroSolver(tjunction_mesh)
    else:
        obj = LondonSolver(tjunction_mesh, Pearllength=1.0)
    with pytest.raises(ValueError):
        obj.solve([0.0, 1.0])
