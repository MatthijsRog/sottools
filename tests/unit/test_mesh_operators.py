import numpy as np
import pytest

from helpers import assert_zero


def test_p1_basis(bar_mesh):
    """The function f = a + b*x + c*y should have an exact representation
    in the P1 basis, and the coefficients should be recoverable from the
    operators gradient_operator_x, gradient_operator_y."""

    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(-5, 5, size=3)

    verts = bar_mesh.vertices
    f = a + b * verts[:, 0] + c * verts[:, 1]

    np.testing.assert_allclose(
        bar_mesh.gradient_operator_x @ f, b * np.ones(len(bar_mesh.triangles))
    )
    np.testing.assert_allclose(
        bar_mesh.gradient_operator_y @ f, c * np.ones(len(bar_mesh.triangles))
    )


def test_field_curl(bar_mesh):
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(-5, 5, size=3)
    verts = bar_mesh.vertices
    f = a + b * verts[:, 0] + c * verts[:, 1]

    Kx, Ky = bar_mesh.field_curl(f)
    Kx_direct, Ky_direct = (
        bar_mesh.gradient_operator_y @ f,
        -bar_mesh.gradient_operator_x @ f,
    )

    np.testing.assert_allclose(Kx, Kx_direct)
    np.testing.assert_allclose(Ky, Ky_direct)


def test_shapegradients(bar_mesh):
    # Sum of shape gradients for a triangle should be zero
    for i, _ in enumerate(bar_mesh.triangles):
        shapegrads = bar_mesh.shapegradients[i]
        sum_shapegrads = np.sum(shapegrads, axis=0)
        assert_zero(sum_shapegrads)


def test_constant_field_gradient(bar_mesh):
    # The gradient of a constant field should be zero
    constant_field = np.ones(len(bar_mesh.vertices))
    grad_x = bar_mesh.gradient_operator_x @ constant_field
    grad_y = bar_mesh.gradient_operator_y @ constant_field

    assert_zero(grad_x)
    assert_zero(grad_y)


def test_edge_difference_operator(bar_mesh):
    edge_diff = bar_mesh.edgedifference_operator
    n_edges, n_triangles = edge_diff.shape
    assert n_triangles == len(bar_mesh.triangles)

    # Exactly one +1 / one -1 per row
    assert edge_diff.nnz == 2 * n_edges
    np.testing.assert_array_equal(np.unique(edge_diff.data), [-1, 1])

    # Rows cancel: constant per-triangle field maps to zero
    row_sums = np.asarray(edge_diff.sum(axis=1)).ravel()
    assert_zero(row_sums)

    constant_field = np.ones(n_triangles)  # per-triangle, not per-vertex
    assert_zero(edge_diff @ constant_field)


def test_stiffness_matrix(bar_mesh):
    # Stiffness matrix should be symmetric
    mat = bar_mesh.stiffness_matrix
    assert (mat != mat.T).nnz == 0

    # Constant functions should lie in the nullspace
    const = np.ones(len(bar_mesh.vertices))
    assert_zero(mat @ const)

    # |∇f|²·area should equal fᵀ·K·f for linear functions f
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(-5, 5, size=3)
    verts = bar_mesh.vertices
    areas = bar_mesh.areas
    f = a + b * verts[:, 0] + c * verts[:, 1]

    matcalc = f @ (mat @ f)
    manual_calc = (b**2 + c**2) * areas.sum()

    np.testing.assert_allclose(matcalc, manual_calc)

    # Shape should be (N_v, N_v)
    assert mat.shape == (len(bar_mesh.vertices), len(bar_mesh.vertices))


def test_mass_matrix(bar_mesh):
    # Mass matrix should be symmetric
    mat = bar_mesh.mass_matrix
    assert (mat != mat.T).nnz == 0

    # Multiplying with unity should yield total area
    const = np.ones(len(bar_mesh.vertices))
    assert const @ (mat @ const) == pytest.approx(bar_mesh.areas.sum())

    # For linear functions, fᵀ·M·f should equal ∫ f² dA
    rng = np.random.default_rng(0)
    a, b, c = rng.uniform(-5, 5, size=3)
    verts = bar_mesh.vertices
    areas = bar_mesh.areas
    f = a + b * verts[:, 0] + c * verts[:, 1]

    matcalc = f @ (mat @ f)

    tri = bar_mesh.triangles
    ft = f[tri]  # (N_t, 3)
    ref = np.sum(areas / 12 * (np.sum(ft**2, axis=1) + np.sum(ft, axis=1) ** 2))
    np.testing.assert_allclose(matcalc, ref)

    # Shape should be (N_v, N_v)
    assert mat.shape == (len(bar_mesh.vertices), len(bar_mesh.vertices))
