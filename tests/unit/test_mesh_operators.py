import numpy as np
import pytest
import pickle

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

def _circumcentres(mesh) -> np.ndarray:
    """Circumcentres from the perpendicular-bisector formula, independent of B."""
    A, B, C = (mesh.vertices[mesh.triangles[:, k]] for k in range(3))
    d = 2 * (A[:, 0] * (B[:, 1] - C[:, 1]) + B[:, 0] * (C[:, 1] - A[:, 1])
             + C[:, 0] * (A[:, 1] - B[:, 1]))
    a2, b2, c2 = (A**2).sum(1), (B**2).sum(1), (C**2).sum(1)
    ux = (a2 * (B[:, 1] - C[:, 1]) + b2 * (C[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / d
    uy = (a2 * (C[:, 0] - B[:, 0]) + b2 * (A[:, 0] - C[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / d
    return np.column_stack([ux, uy])


def _random_interior_field(mesh, seed: int = 0) -> np.ndarray:
    """Random nodal field, zero on the boundary vertices."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=len(mesh.vertices))
    g[mesh.dirichlet_idx] = 0.0
    return g


def test_node_to_cc_operator_shape(disk_mesh_freestanding):
    m = disk_mesh_freestanding
    assert m.node_to_cc_operator.shape == (len(m.triangles), len(m.vertices))
    assert m.node_to_cc_operator.nnz == 3 * len(m.triangles)


def test_node_to_cc_operator_row_sums(disk_mesh_freestanding):
    # Barycentric coordinates sum to 1; the 16 A^2 denominator makes this non-trivial
    row_sums = np.asarray(disk_mesh_freestanding.node_to_cc_operator.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=1e-12)


def test_node_to_cc_operator_reproduces_circumcentres(disk_mesh_freestanding):
    # x and y are affine, so B evaluates them exactly at the circumcentres
    m = disk_mesh_freestanding
    cc = _circumcentres(m)
    np.testing.assert_allclose(m.node_to_cc_operator @ m.vertices, cc, rtol=0, atol=1e-12)


def test_node_to_cc_operator_matches_shapegradients(disk_mesh_freestanding):
    # Entry for local vertex k of T: lambda_k(c_T) = 1/3 + grad(lambda_k) . (c_T - centroid)
    m = disk_mesh_freestanding
    cc = _circumcentres(m)
    expected = 1 / 3 + np.einsum("tkd,td->tk", m.shapegradients, cc - m.centers)  # (N_t, 3)
    rows = np.arange(len(m.triangles))[:, None].repeat(3, axis=1)
    entries = np.asarray(m.node_to_cc_operator[rows.ravel(), m.triangles.ravel()]).ravel()
    np.testing.assert_allclose(entries, expected.ravel(), rtol=0, atol=1e-10)


def test_node_to_cc_operator_covers_obtuse_triangles(disk_mesh_freestanding):
    # Otherwise a sign error confined to obtuse triangles goes unnoticed
    assert (disk_mesh_freestanding.node_to_cc_operator.data < 0).any()


def test_nodes_from_triangles_roundtrip(disk_mesh_freestanding):
    # node -> triangle -> node is the identity on fields that vanish on the boundary
    m = disk_mesh_freestanding
    g = _random_interior_field(m)
    g_back = m.nodes_from_triangles(m.node_to_cc_operator @ g)
    np.testing.assert_allclose(g_back, g, rtol=0, atol=1e-9 * np.abs(g).max())


def test_nodes_from_triangles_zero_on_boundary(disk_mesh_freestanding):
    m = disk_mesh_freestanding
    rng = np.random.default_rng(1)
    g = m.nodes_from_triangles(rng.normal(size=len(m.triangles)))
    assert g.shape == (len(m.vertices),)
    np.testing.assert_array_equal(g[m.dirichlet_idx], 0.0)


def test_nodes_from_triangles_is_least_squares(disk_mesh_freestanding):
    # Normal equations: the residual is orthogonal to every free column of B
    m = disk_mesh_freestanding
    rng = np.random.default_rng(2)
    g_T = rng.normal(size=len(m.triangles))   # generic, not in the range of B
    g = m.nodes_from_triangles(g_T)
    B_free = m.node_to_cc_operator[:, m.free_idx]
    residual = g_T - m.node_to_cc_operator @ g
    np.testing.assert_allclose(B_free.T @ residual, 0.0, rtol=0, atol=1e-9 * np.abs(g_T).max())

def test_freestanding_mesh_pickle_roundtrip(disk_mesh_freestanding):
    # Both cached SuperLU factorizations must survive pickling
    m = disk_mesh_freestanding
    m2 = pickle.loads(pickle.dumps(m))
    rng = np.random.default_rng(3)
    g_T = rng.normal(size=len(m.triangles))
    np.testing.assert_allclose(m2.nodes_from_triangles(g_T), m.nodes_from_triangles(g_T),
                               rtol=0, atol=1e-12)
    rhs = rng.normal(size=len(m.free_idx))
    np.testing.assert_allclose(m2.stiffness_matrix_solver.solve(rhs),
                               m.stiffness_matrix_solver.solve(rhs), rtol=0, atol=1e-12)

def test_edge_flux_operator_structure(disk_mesh_freestanding):
    m = disk_mesh_freestanding
    F = m.edge_flux_operator
    assert F.shape == (len(m.edges), len(m.triangles))
    assert F.nnz == 3 * len(m.triangles)
    entries_per_edge = np.diff(F.indptr)
    np.testing.assert_array_equal(entries_per_edge[m.interior_edge_mask], 2)
    np.testing.assert_array_equal(entries_per_edge[m.boundary_edge_mask], 1)
    # Interior rows: one +1 and one -1, so a constant field has zero flux there
    const_flux = F @ np.ones(len(m.triangles))
    np.testing.assert_array_equal(const_flux[m.interior_edge_mask], 0.0)


def test_edge_flux_operator_linear_stream_function(disk_mesh_freestanding):
    # g = a . r gives K = (a_y, -a_x); flux through an interior edge = s_ij K . e_hat
    m = disk_mesh_freestanding
    a = np.array([0.7, -1.3])
    cc = _circumcentres(m)
    flux = m.edge_flux_operator @ (cc @ a)
    K = np.array([a[1], -a[0]])
    e_hat = m.edge_vectors / m.edge_lengths[:, None]
    s = m.cotangent_weights * m.edge_lengths  # signed dual-edge lengths
    expected = s * (e_hat @ K)
    interior = m.interior_edge_mask
    np.testing.assert_allclose(flux[interior], expected[interior], rtol=0,
                               atol=1e-12 * np.abs(expected).max())


def test_edge_divergence_operator_outflow_positive(disk_mesh_freestanding):
    # A unit flux on edge (i, j) leaves i and enters j
    m = disk_mesh_freestanding
    Dv = m.edge_divergence_operator
    assert Dv.shape == (len(m.vertices), len(m.edges))
    e = len(m.edges) // 2
    unit = np.zeros(len(m.edges))
    unit[e] = 1.0
    div = Dv @ unit
    i, j = m.edges[e]
    assert div[i] == 1.0 and div[j] == -1.0
    assert np.count_nonzero(div) == 2


def test_stream_function_fluxes_are_conserved(disk_mesh_freestanding):
    # Divergence of stream-function fluxes vanishes identically, boundary vertices included
    m = disk_mesh_freestanding
    product = (m.edge_divergence_operator @ m.edge_flux_operator).tocoo()
    assert np.all(product.data == 0.0)

def test_edge_flux_operator_structure(disk_mesh_freestanding):
    m = disk_mesh_freestanding
    F = m.edge_flux_operator
    assert F.shape == (len(m.edges), len(m.triangles))
    assert F.nnz == 3 * len(m.triangles)
    entries_per_edge = np.diff(F.indptr)
    np.testing.assert_array_equal(entries_per_edge[m.interior_edge_mask], 2)
    np.testing.assert_array_equal(entries_per_edge[m.boundary_edge_mask], 1)
    # Interior rows: one +1 and one -1, so a constant field has zero flux there
    const_flux = F @ np.ones(len(m.triangles))
    np.testing.assert_array_equal(const_flux[m.interior_edge_mask], 0.0)


def test_edge_flux_operator_linear_stream_function(disk_mesh_freestanding):
    # g = a . r gives K = (a_y, -a_x); flux through an interior edge = s_ij K . e_hat
    m = disk_mesh_freestanding
    a = np.array([0.7, -1.3])
    cc = _circumcentres(m)
    flux = m.edge_flux_operator @ (cc @ a)
    K = np.array([a[1], -a[0]])
    e_hat = m.edge_vectors / m.edge_lengths[:, None]
    s = m.cotangent_weights * m.edge_lengths  # signed dual-edge lengths
    expected = s * (e_hat @ K)
    interior = m.interior_edge_mask
    np.testing.assert_allclose(flux[interior], expected[interior], rtol=0,
                               atol=1e-12 * np.abs(expected).max())


def test_edge_divergence_operator_outflow_positive(disk_mesh_freestanding):
    # A unit flux on edge (i, j) leaves i and enters j
    m = disk_mesh_freestanding
    Dv = m.edge_divergence_operator
    assert Dv.shape == (len(m.vertices), len(m.edges))
    e = len(m.edges) // 2
    unit = np.zeros(len(m.edges))
    unit[e] = 1.0
    div = Dv @ unit
    i, j = m.edges[e]
    assert div[i] == 1.0 and div[j] == -1.0
    assert np.count_nonzero(div) == 2


def test_stream_function_fluxes_are_conserved(disk_mesh_freestanding):
    # Divergence of stream-function fluxes vanishes identically, boundary vertices included
    m = disk_mesh_freestanding
    product = (m.edge_divergence_operator @ m.edge_flux_operator).tocoo()
    assert np.all(product.data == 0.0)