import numpy as np

import sottools.mesh as Mesh


def _on_any_segment(points, verts, segs, tol=1e-9):
    """Boolean mask: which points lie on any of the given segments."""
    on = np.zeros(len(points), dtype=bool)
    for a, b in segs:
        A = np.asarray(verts[a], float)
        B = np.asarray(verts[b], float)
        AB = B - A
        L2 = AB @ AB
        AP = points - A  # (N, 2)
        cross = AB[0] * AP[:, 1] - AB[1] * AP[:, 0]
        t = (AP @ AB) / L2
        on |= (np.abs(cross) < tol * np.sqrt(L2)) & (t >= -tol) & (t <= 1 + tol)
    return on


def test_dirichlet_masks(complex_mesh_input):
    vertices, segments, segment_markers = complex_mesh_input
    mesh = Mesh.SimplyConnectedCurrentMesh(
        vertices,
        segments,
        segment_markers,
        max_area=0.1,
        min_angle=30.0,
        refine_distance=0.05,
    )
    BM = Mesh.SimplyConnectedCurrentMesh.BoundaryMarker

    left_segs = [
        s
        for s, m in zip(segments, segment_markers, strict=True)
        if m == BM.DIRICHLET_LEFT
    ]
    right_segs = [
        s
        for s, m in zip(segments, segment_markers, strict=True)
        if m == BM.DIRICHLET_RIGHT
    ]

    expect_left = _on_any_segment(mesh.vertices, vertices, left_segs)
    expect_right = _on_any_segment(mesh.vertices, vertices, right_segs)

    # The dirichlet masks should correspond to what we expect from the original
    # segment input
    np.testing.assert_array_equal(mesh.dirichletleft_mask, expect_left)
    np.testing.assert_array_equal(mesh.dirichletright_mask, expect_right)

    # Partition/disjointness properties (independent of geometry)
    assert not (mesh.dirichletleft_mask & mesh.dirichletright_mask).any()
    np.testing.assert_array_equal(mesh.free_mask, ~mesh.dirichlet_mask)
    assert mesh.free_idx.size + mesh.dirichlet_idx.size == len(mesh.vertices)
