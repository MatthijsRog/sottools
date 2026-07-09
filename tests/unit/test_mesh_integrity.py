import numpy as np
import pytest

import sottools.mesh as Mesh


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
