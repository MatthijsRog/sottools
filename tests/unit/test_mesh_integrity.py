import pickle

import numpy as np
import pytest
import scipy.sparse

import sottools.mesh as Mesh

# Attributes carried by pickles made with sottools 0.2.0
_STATE_KEYS_0_2_0 = (
    "_vertices",
    "_triangles",
    "_neighbors",
    "_segments",
    "_segment_markers",
    "_boundary_vertices",
    "_areas",
    "_centers",
    "_shapegradients",
    "_dirichletleft_mask",
    "_dirichletright_mask",
    "_dirichlet_mask",
    "_free_mask",
    "_free_idx",
    "_dirichlet_idx",
    "_internaledgelengths",
    "_edgedifference_operator",
    "_gradient_operator_x",
    "_gradient_operator_y",
    "_stiffness_matrix",
    "_mass_matrix",
)

BM = Mesh.SimplyConnectedCurrentMesh.BoundaryMarker


def _assert_same_state(a, b):
    state_a, state_b = a.__getstate__(), b.__getstate__()
    assert state_a.keys() == state_b.keys()
    for key, value in state_a.items():
        if scipy.sparse.issparse(value):
            assert (value != state_b[key]).nnz == 0, key
        else:
            np.testing.assert_array_equal(value, state_b[key], err_msg=key)

    # The SuperLU factorization is not in the state; check it was rebuilt
    rhs = np.random.default_rng(0).standard_normal(a.free_idx.size)
    np.testing.assert_array_equal(
        a.stiffness_matrix_solver.solve(rhs), b.stiffness_matrix_solver.solve(rhs)
    )


def test_area_sum_equals_polygon_area(bar_mesh):
    assert bar_mesh.areas.sum() == pytest.approx(24.0)


def test_triangle_areas_positive(bar_mesh):
    assert (bar_mesh.areas > 0).all()


def test_cell_centers_inside_triangles(bar_mesh):
    # A cell center is inside the triangle if all barycentric coordinates are
    # between 0 and 1
    for i, triangle in enumerate(bar_mesh.triangles):
        vertices = bar_mesh.vertices[triangle]
        A = vertices[0]
        B = vertices[1]
        C = vertices[2]
        P = bar_mesh.centers[i]

        # Compute barycentric coordinates
        v0 = B - A
        v1 = C - A
        v2 = P - A

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        assert 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1


def test_refinement(complex_mesh_input):
    vertices, segments, segment_markers = complex_mesh_input
    refine_distance = 0.05
    mesh = Mesh.SimplyConnectedCurrentMesh(
        vertices,
        segments,
        segment_markers,
        max_area=0.1,
        min_angle=30.0,
        verbose=False,
        refine_distance=refine_distance,
    )

    # Check that all boundary edges are smaller than refine_distance
    for seg in mesh.boundary_segments:
        edge_length = np.linalg.norm(mesh.vertices[seg[0]] - mesh.vertices[seg[1]])
        assert edge_length <= refine_distance or edge_length == pytest.approx(
            refine_distance
        )

    # Furthermore, validate structure: all original vertices should be present
    # in the refined mesh
    for v in vertices:
        assert any(np.allclose(v, mv) for mv in mesh.vertices)


def test_mesh_input_validation(complex_mesh_input):
    vertices, segments, segment_markers = complex_mesh_input

    # Test length mismatches
    segment_markers_truncated = segment_markers[:-1]
    with pytest.raises(ValueError):
        Mesh.SimplyConnectedCurrentMesh(
            vertices, segments, segment_markers_truncated, max_area=0.1
        )


def test_boundary_markers_pinned(complex_mesh_input):
    """This test confirms that NEUMANN_IN, NEUMANN_OUT, DIRICHLET_IN and DIRICHLET_OUT
    markers continue to work, and still follow:

    DIRICHLET_LEFT = 1
    DIRICHLET_RIGHT = 2
    NEUMANN_IN = 3
    NEUMANN_OUT = 4

    For sottools v1.0.0, we will most likely release this criterion and the test should
    then be removed.
    """
    # First check that the boundary markers are still pinned to the correct values
    assert Mesh.SimplyConnectedCurrentMesh.BoundaryMarker.DIRICHLET_LEFT == 1
    assert Mesh.SimplyConnectedCurrentMesh.BoundaryMarker.DIRICHLET_RIGHT == 2
    assert Mesh.SimplyConnectedCurrentMesh.BoundaryMarker.NEUMANN_IN == 3
    assert Mesh.SimplyConnectedCurrentMesh.BoundaryMarker.NEUMANN_OUT == 4

    # Now compare a mesh made from enums with a mesh made from integers
    # which should be identical in all masks, their diriclet_idx and stiffness matrix
    vertices, segments, segment_markers = complex_mesh_input
    segment_markers_int = [
        3,
        2,
        4,
        1,
        1,
        1,
        1,
        1,
    ]
    mesh_from_enums = Mesh.SimplyConnectedCurrentMesh(
        vertices,
        segments,
        segment_markers,
        max_area=0.1,
        min_angle=30.0,
        refine_distance=0.05,
    )
    mesh_from_ints = Mesh.SimplyConnectedCurrentMesh(
        vertices,
        segments,
        segment_markers_int,
        max_area=0.1,
        min_angle=30.0,
        refine_distance=0.05,
    )
    assert np.array_equal(mesh_from_enums.dirichlet_mask, mesh_from_ints.dirichlet_mask)
    assert np.array_equal(mesh_from_enums.dirichlet_idx, mesh_from_ints.dirichlet_idx)
    assert np.array_equal(
        mesh_from_enums.stiffness_matrix.todense(),
        mesh_from_ints.stiffness_matrix.todense(),
    )


def test_pickle_roundtrip(bar_mesh):
    restored = pickle.loads(pickle.dumps(bar_mesh))
    _assert_same_state(bar_mesh, restored)


def test_wall_markers():
    assert BM.WALL(0) is BM.DIRICHLET_LEFT
    assert BM.WALL(1) is BM.DIRICHLET_RIGHT
    assert [BM.wall_index(BM.WALL(k)) for k in range(100)] == list(range(100))
    assert BM.wall_index(BM.NEUMANN_IN) is None
    assert BM.wall_index(BM.NEUMANN_OUT) is None
    with pytest.raises(ValueError):
        BM.WALL(100)


def test_wall_order(tjunction_input):
    vertices, segments, markers = tjunction_input
    order = Mesh.SimplyConnectedCurrentMesh._boundary_wall_order

    np.testing.assert_array_equal(order(vertices, segments, markers), [0, 1, 2])

    # Clockwise input gives the same counter-clockwise order
    reversed_segments = [[b, a] for a, b in segments]
    np.testing.assert_array_equal(
        order(vertices, reversed_segments, markers), [0, 1, 2]
    )

    # Swapping the labels of walls 1 and 2 swaps them in the order
    swap = {BM.WALL(1): BM.WALL(2), BM.WALL(2): BM.WALL(1)}
    swapped = [swap.get(m, m) for m in markers]
    np.testing.assert_array_equal(order(vertices, segments, swapped), [0, 2, 1])


@pytest.mark.parametrize(
    "overrides",
    [
        {0: BM.INTERIOR},
        {2: BM.NEUMANN_IN, 3: BM.NEUMANN_IN},
        {4: BM.WALL(1)},
        {5: BM.WALL(0), 6: BM.WALL(0)},
        {5: BM.WALL(3), 6: BM.WALL(3)},
    ],
    ids=[
        "interior_marker",
        "adjacent_contacts",
        "adjacent_walls",
        "wall_on_two_chains",
        "wall_index_gap",
    ],
)
def test_invalid_boundary_rejected(overrides, tjunction_input):
    vertices, segments, markers = tjunction_input
    bad = [overrides.get(s, m) for s, m in enumerate(markers)]
    with pytest.raises(ValueError):
        Mesh.SimplyConnectedCurrentMesh(vertices, segments, bad)


def test_open_boundary_rejected(tjunction_input):
    vertices, segments, markers = tjunction_input
    with pytest.raises(ValueError):
        Mesh.SimplyConnectedCurrentMesh(vertices, segments[:-1], markers[:-1])
