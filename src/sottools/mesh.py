"""2D FEM mesh tools and solvers for basic physics on them.

This module contains classes and functions for creating and manipulating
2D meshes. Currently, we exclusively support simply connected geometries that
carry current. The LondonSolver class implements the simulation of
superconducting current flow in these meshes using the London equations.

For plotting functions, we refer to the `Meshplotting` module
"""

import abc
import enum
import time
import typing

import numpy as np
import scipy
import triangle


class SingleBodyMesh(abc.ABC):
    """Base class for FEM meshes that consist of a single body."""

    @property
    @abc.abstractmethod
    def vertices(self) -> np.ndarray:
        """Vertices of the mesh.

        Returns a (N_v,2) array of vertex coordinates.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def triangles(self) -> np.ndarray:
        """Triangulation of the mesh.

        Returns a (N_t,3) array containing the vertex indices of a triangle.
        """
        raise NotImplementedError


class SimplyConnectedCurrentMesh(SingleBodyMesh):
    """2D FEM mesh for simply connected current-carrying geometries.

    A 2D FEM mesh catered to simply connected (hole-free) geometries that
    carry current, and therefore have well-defined sources and sinks.

    Pre-defines basis functions, gradients and matrices such that FEM problems
    and calculations can easily be set up.

    Parameters
    ----------
    vertices : list[list[float]]
        List of 2D coordinates. These are the outer points of the geometry.
    segments: list[list[int]]
        List of vertex index pairs defining which vertices are connected.
    segment_markers: list[int]
        List of the same length as segments. Each entry defines the boundary
        type of that segment. The left and right sides are called
        .DIRICHLET_LEFT and .DIRICHLET_RIGHT. The inlet and outlet are called
        .NEUMANN_IN and .NEUMANN_OUT.
    max_area: float
        Maximum area of a single mesh cell, enforced in the triangulation.
        Defaults to 0.1
    min_angle: float
        Minimum angle of a single mesh cell, enforced in the triangulation.
        Defaults to 30.0 degrees.
    verbose: bool
        If true, debug messages are printed to console. Defaults to False.
    refine_distance: float or None
        If not none, the outer boundary gets subdivided until the smallest
        boundary segement has length refine_distance. Essential if a fine
        meshing at the edge, e.g. for London problems, is desired.
    """

    class BoundaryMarker(enum.IntEnum):
        """The various types a boundary segment can have."""

        INTERIOR = 0
        DIRICHLET_LEFT = 1
        DIRICHLET_RIGHT = 2
        NEUMANN_IN = 3
        NEUMANN_OUT = 4

    def __init__(
        self,
        vertices: list[list[float]],
        segments: list[list[int]],
        segment_markers: list[BoundaryMarker],
        max_area: float = 0.1,
        min_angle: float = 30.0,
        verbose: bool = False,
        refine_distance: float | None = None,
    ):
        if verbose:
            print("Triangulating mesh...")
            start_time = time.time()

        if len(segments) != len(segment_markers):
            raise ValueError(
                f"segments ({len(segments)}) and segment_markers "
                f"({len(segment_markers)}) must have the same length"
            )

        # Refine
        if refine_distance is not None:
            fine_mesh_data = self._refine_boundary(
                vertices, segments, segment_markers, refine_distance
            )
        else:
            fine_mesh_data = (
                vertices,
                segments,
                segment_markers,
            )
        vertices_fine, segments_fine, segment_markers_fine = fine_mesh_data

        meshdict = triangle.triangulate(
            {
                "vertices": vertices_fine,
                "segments": segments_fine,
                "segment_markers": segment_markers_fine,
            },
            f"pq{min_angle}a{max_area}n",
        )

        if verbose:
            print(f"Triangulation completed in {time.time() - start_time:.2f} seconds")
            print("Next step: calculating matrices...")
            start_time = time.time()

        self._vertices: np.ndarray = meshdict["vertices"]
        self._triangles: np.ndarray = meshdict["triangles"]
        self._neighbors: np.ndarray = meshdict["neighbors"]
        self._segments: np.ndarray = meshdict["segments"]
        self._segment_markers = meshdict["segment_markers"].ravel()
        self._boundary_vertices = np.array(vertices)

        # Cell areas
        self._areas = self._cell_areas()
        assert (self._areas > 0).all(), "CW triangle(s) found"

        # Cell centers
        self._centers = self._cell_centers()

        # Cell gradients per vertex
        self._shapegradients = self._computeshapegradients()
        assert np.allclose(self._shapegradients.sum(axis=1), 0)

        # Find masks
        self._dirichletleft_mask, self._dirichletright_mask, self._dirichlet_mask = (
            self._dirichletmasks()
        )
        self._free_mask = ~self._dirichlet_mask
        self._free_idx = np.where(self._free_mask)[0]
        self._dirichlet_idx = np.where(self._dirichlet_mask)[0]

        # Cache matrices and operators
        self._internaledgelengths, self._edgedifference_operator = (
            self._edgedifferenceoperator()
        )
        self._gradient_operator_x, self._gradient_operator_y = self._gradientoperators(
            self._shapegradients
        )
        self._stiffness_matrix = self._stiffnessmatrix(self._shapegradients)
        self._stiffness_matrix_solver = self._freesolver(
            self.stiffness_matrix, self.free_idx
        )  # Caches the (N_free, N_free) part of this matrix for quick solving
        self._mass_matrix = self._massmatrix()

        if verbose:
            N_v = self.vertices.shape[0]
            N_t = self.triangles.shape[0]

            print(f"Matrices calculated in {time.time() - start_time:.2f} seconds")
            print("Number of vertices:", N_v)
            print("Number of triangles:", N_t)

    def __getstate__(self) -> dict[str, object]:
        """Conversion of object to dictionary.

        Special care must be taken here because SuperLU objects cannot be
        converted to a dictionary, and can then not be pickled.

        Returns
        -------
        state: dict
            Dictionary representation of the object.
        """
        state = self.__dict__.copy()
        del state["_stiffness_matrix_solver"]  # SuperLU can't be pickled
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Conversion of dictionary to object.

        Special care must be taken here because SuperLU objects cannot be
        converted to a dictionary, and can then not be pickled.

        Parameters
        ----------
        state: dict
            Dictionary representation of the object.
        """
        self.__dict__.update(state)
        # Recompute the factorization
        self._stiffness_matrix_solver = self._freesolver(
            self.stiffness_matrix, self.free_idx
        )

    @staticmethod
    def _refine_boundary(
        vertices: list[list[float]],
        segments: list[list[int]],
        segment_markers: list[BoundaryMarker],
        edge_spacing: float = 0.05,
    ) -> tuple[list[list[float]], list[list[int]], list[BoundaryMarker]]:
        """Refines outer boundary of the mesh.

        Parameters
        ----------
        vertices: list[list[float]]
            List of 2D coordinates. These are the outer points of the geometry.
        segments: list[list[int]]
            List of vertex index pairs defining which vertices are connected.
        segment_markers: list[int]
            List of the same length as segments. Each entry defines the
            boundary type of that segment. The left and right sides are called
            .DIRICHLET_LEFT and .DIRICHLET_RIGHT. The inlet and outlet are
            called .NEUMANN_IN and .NEUMANN_OUT.
        edge_spacing: float
            Desired spacing between vertices on the boundary. Defaults to 0.05.
        """
        new_vertices = list(vertices)
        new_segments = []
        new_markers = []

        for seg, marker in zip(segments, segment_markers, strict=True):
            p0 = np.array(vertices[seg[0]], dtype=float)
            p1 = np.array(vertices[seg[1]], dtype=float)
            length = np.linalg.norm(p1 - p0)
            n_subdivisions = max(1, int(np.ceil(length / edge_spacing)))

            idx_start = seg[0]
            for i in range(1, n_subdivisions):
                t = i / n_subdivisions
                new_point = (1 - t) * p0 + t * p1
                idx_new = len(new_vertices)
                new_vertices.append(new_point.tolist())
                new_segments.append([idx_start, idx_new])
                new_markers.append(marker)
                idx_start = idx_new

            new_segments.append([idx_start, seg[1]])
            new_markers.append(marker)

        return new_vertices, new_segments, new_markers

    def _cell_areas(self) -> np.ndarray:
        """Signed areas: positive for counter-clockwise oriented triangles."""
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        bs = v1s - v0s
        cs = v2s - v0s
        areas: np.ndarray = 0.5 * (bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])

        return areas

    def _cell_centers(self) -> np.ndarray:
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        centers: np.ndarray = (v0s + v1s + v2s) / 3
        return centers

    def _computeshapegradients(self) -> np.ndarray:
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        b = np.column_stack(
            [v1s[:, 1] - v2s[:, 1], v2s[:, 1] - v0s[:, 1], v0s[:, 1] - v1s[:, 1]]
        )  # (N_t, 3)
        c = np.column_stack(
            [v2s[:, 0] - v1s[:, 0], v0s[:, 0] - v2s[:, 0], v1s[:, 0] - v0s[:, 0]]
        )  # (N_t, 3)

        shapegrads: np.ndarray = (
            np.stack([b, c], axis=-1) / (2 * self._areas)[:, None, None]
        )
        return shapegrads

    def _dirichletmasks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N_v = self.vertices.shape[0]

        dirichlet_left_mask = np.zeros(N_v, dtype=bool)
        dirichlet_right_mask = np.zeros(N_v, dtype=bool)

        for marker, mask in [
            (self.BoundaryMarker.DIRICHLET_RIGHT, dirichlet_right_mask),
            (self.BoundaryMarker.DIRICHLET_LEFT, dirichlet_left_mask),
        ]:
            seg_mask = self._segment_markers == marker
            verts = np.unique(self._segments[seg_mask].ravel())
            mask[verts] = True

        dirichlet_mask = dirichlet_left_mask | dirichlet_right_mask

        return dirichlet_left_mask, dirichlet_right_mask, dirichlet_mask

    def _edgedifferenceoperator(self) -> tuple[np.ndarray, scipy.sparse.csr_matrix]:
        internal_edges = []  # (t1, t2) triangle pairs where t1 and t2 share an edge
        internal_edge_vertices = []  # (v1, v2) vertex pairs for each internal edge
        boundary_edges = []  # (v1, v2) vertex pairs for each boundary edge

        for t, tri in enumerate(self.triangles):
            for i in range(3):
                v1, v2 = tri[(i + 1) % 3], tri[(i + 2) % 3]
                neighbor = self.neighbors[t, i]
                if neighbor != -1:
                    if t < neighbor:  # Avoid double counting
                        internal_edges.append((t, neighbor))
                        internal_edge_vertices.append((v1, v2))
                else:
                    boundary_edges.append((v1, v2))

        (internal_edges_arr, internal_edge_vertices_arr) = (
            np.array(internal_edges),
            np.array(internal_edge_vertices),
        )

        internal_edge_lengths = np.linalg.norm(
            self.vertices[internal_edge_vertices_arr[:, 0]]
            - self.vertices[internal_edge_vertices_arr[:, 1]],
            axis=1,
        )

        N_t = self.triangles.shape[0]
        N_e = len(internal_edges)

        # Original implementation:
        # edge_difference_operator = scipy.sparse.lil_matrix((len(internal_edges), N_t))
        # for e, (t1, t2) in enumerate(internal_edges):
        #     edge_difference_operator[e, t1] = 1
        #     edge_difference_operator[e, t2] = -1
        # edge_difference_operator = edge_difference_operator.tocsr()

        rows = np.arange(N_e)[:, None].repeat(2, axis=1)  # (N_e, 2)
        cols = internal_edges_arr.ravel()  # (N_e, 2)
        edge_difference_operator = scipy.sparse.coo_matrix(
            (np.array([1, -1] * N_e), (rows.ravel(), cols.ravel())), shape=(N_e, N_t)
        ).tocsr()

        return internal_edge_lengths, edge_difference_operator

    def _gradientoperators(
        self, shapegradients: np.ndarray
    ) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
        N_v = self.vertices.shape[0]
        N_t = self.triangles.shape[0]

        # Original implementation:
        # gradient_operator_x = scipy.sparse.lil_matrix((N_t, N_v))
        # gradient_operator_y = scipy.sparse.lil_matrix((N_t, N_v))
        # for t in range(N_t):
        #     for i in range(3):
        #         gradient_operator_x[t, self.triangles[t, i]] = shapegradients[t, i, 0]
        #         gradient_operator_y[t, self.triangles[t, i]] = shapegradients[t, i, 1]
        # gradient_operator_x = gradient_operator_x.tocsr()
        # gradient_operator_y = gradient_operator_y.tocsr()

        tri_idx = np.arange(N_t)[:, None].repeat(3, axis=1)  # (N_t, 3)
        ver_idx = self.triangles  # (N_t, 3)

        gradient_operator_x = scipy.sparse.coo_matrix(
            (shapegradients[:, :, 0].ravel(), (tri_idx.ravel(), ver_idx.ravel())),
            shape=(N_t, N_v),
        ).tocsr()
        gradient_operator_y = scipy.sparse.coo_matrix(
            (shapegradients[:, :, 1].ravel(), (tri_idx.ravel(), ver_idx.ravel())),
            shape=(N_t, N_v),
        ).tocsr()

        return gradient_operator_x, gradient_operator_y

    def _stiffnessmatrix(self, shapegradients: np.ndarray) -> scipy.sparse.csr_matrix:
        N_v = self.vertices.shape[0]

        # Original implementation:
        # stiffness_matrix = scipy.sparse.lil_matrix((N_v, N_v))
        # for t in range(N_t):
        #    nodes = self.triangles[t]
        #    for i in range(3):
        #        for j in range(3):
        #            stiffness_matrix[nodes[i], nodes[j]] += self._areas[t] * np.dot(
        #                shapegradients[t, i], shapegradients[t, j]
        #            )
        # stiffness_matrix = stiffness_matrix.tocsr()

        K_t = np.einsum("tiu,tju->tij", shapegradients, shapegradients)  # (N_t, 3, 3)
        K_t *= self._areas[:, None, None]  # Scale by area
        rows = self.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        cols = self.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        K = scipy.sparse.coo_matrix(
            (K_t.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()

        return K

    def _freesolver(
        self, matrix: scipy.sparse.csr_matrix, free_idx: np.ndarray
    ) -> scipy.sparse.linalg.SuperLU:
        matrix_free = matrix[free_idx][:, free_idx]
        solver = scipy.sparse.linalg.splu(matrix_free.tocsc())
        return solver

    def _massmatrix(self) -> scipy.sparse.csr_matrix:
        N_v = self.vertices.shape[0]

        # Original implementation:
        # mass_matrix = scipy.sparse.lil_matrix((N_v, N_v))
        # for t in range(N_t):
        #    nodes = self.triangles[t]
        #    for i in range(3):
        #        for j in range(3):
        #            mass_matrix[nodes[i], nodes[j]] += self._areas[t] * (
        #                1 / 12 if i != j else 1 / 6
        #            )
        # mass_matrix = mass_matrix.tocsr()

        local_M = (
            np.where(
                np.eye(3, dtype=bool)[None, :, :],  # (1,3,3)
                1 / 6,  # diagonal entries
                1 / 12,  # off-diagonal entries
            )
            * self._areas[:, None, None]
        )  # (N_t, 3, 3)

        rows = self.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        cols = self.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        M = scipy.sparse.coo_matrix(
            (local_M.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()

        return M

    def field_curl(
        self,
        field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the curl of a field defined on the vertices.

        Represents the operation v = nabla x (f zhat), where the field f
        is a scalar field defined on the vertices. This is exactly how
        streamfunctions are related to sheet current densities in 2D.

        Parameters
        ----------
        field: np.ndarray
            Field defined on the vertices of the mesh. Shape (N_v,).

        Returns
        -------
        Kx: np.ndarray
            x-component of the curl of the field, defined on the cell
            centers. Shape (N_t,).
        Ky: np.ndarray
            y-component of the curl of the field, defined on the cell
            centers. Shape (N_t,).
        """
        field_ts = field[self.triangles]
        grad_field = (field_ts[:, :, None] * self._shapegradients).sum(axis=1)
        Kx = grad_field[:, 1]
        Ky = -grad_field[:, 0]
        return Kx, Ky

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the mesh.

        Returns a (N_v,2) array of vertex coordinates.
        """
        return self._vertices

    @property
    def triangles(self) -> np.ndarray:
        """Triangulation of the mesh.

        Returns a (N_t,3) array containing the vertex indices of a triangle.
        """
        return self._triangles

    @property
    def areas(self) -> np.ndarray:
        """Areas of the triangles in the mesh.

        Returns a (N_t,) array containing the areas of the triangles.
        """
        return self._areas

    @property
    def centers(self) -> np.ndarray:
        """Centers of the triangles in the mesh.

        Returns a (N_t,2) array containing the coordinates of the triangle centers.
        """
        return self._centers

    @property
    def neighbors(self) -> np.ndarray:
        """Neighbor relationships between triangles.

        Returns a (N_t,3) array containing the triangle indices of the
        neighbors of a triangle.
        """
        return self._neighbors

    @property
    def boundary_vertices(self) -> np.ndarray:
        """Vertices on the boundary of the mesh.

        Returns a (N_b,2) array containing the coordinates of the boundary vertices.
        """
        return self._boundary_vertices

    @property
    def boundary_segments(self) -> np.ndarray:
        """Boundary segments of the mesh.

        Returns a (N_s, 2) array of vertex index pairs defining boundary edges.
        """
        return self._segments

    @property
    def shapegradients(self) -> np.ndarray:
        """Gradients of the basis functions.

        Returns a (N_t,3,2) array containing the gradients of the P1
        linear basis functions for each triangle.
        """
        return self._shapegradients

    @property
    def dirichletleft_mask(self) -> np.ndarray:
        """Mask for Dirichlet boundary conditions on the left side of the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the boundary.
        """
        return self._dirichletleft_mask

    @property
    def dirichletright_mask(self) -> np.ndarray:
        """Mask for Dirichlet boundary conditions on the right side of the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the boundary.
        """
        return self._dirichletright_mask

    @property
    def dirichlet_mask(self) -> np.ndarray:
        """Mask for all Dirichlet boundary conditions on the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the boundary.
        """
        return self._dirichlet_mask

    @property
    def free_mask(self) -> np.ndarray:
        """Mask for free vertices (not on Dirichlet boundaries) of the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the free vertices.
        """
        return self._free_mask

    @property
    def free_idx(self) -> np.ndarray:
        """Indices of free vertices (not on Dirichlet boundaries) of the mesh."""
        return self._free_idx

    @property
    def dirichlet_idx(self) -> np.ndarray:
        """Indices of Dirichlet vertices (on Dirichlet boundaries) of the mesh."""
        return self._dirichlet_idx

    @property
    def internaledgelengths(self) -> np.ndarray:
        """Lengths of the internal edges of the mesh.

        Returns a (N_e,) array containing the lengths of the internal edges of
        the mesh.
        """
        return self._internaledgelengths

    @property
    def edgedifference_operator(self) -> scipy.sparse.csr_matrix:
        """Operator for the difference of a field across internal.

        This csr matrix represents a matrix of shape (N_e, N_t) that calculates
        the difference of a field across internal edges.
        """
        return self._edgedifference_operator

    @property
    def gradient_operator_x(self) -> scipy.sparse.csr_matrix:
        """Operator for the x-gradient of a field.

        This csr matrix represents a matrix of shape (N_t, N_v) that calculates
        the x-gradient of a field defined on the vertices.
        """
        return self._gradient_operator_x

    @property
    def gradient_operator_y(self) -> scipy.sparse.csr_matrix:
        """Operator for the y-gradient of a field.

        This csr matrix represents a matrix of shape (N_t, N_v) that calculates
        the y-gradient of a field defined on the vertices.
        """
        return self._gradient_operator_y

    @property
    def stiffness_matrix(self) -> scipy.sparse.csr_matrix:
        """Stiffness matrix of the mesh.

        This csr matrix represents a matrix of shape (N_v, N_v) that calculates
        the stiffness, where the (i,j) component is the integral of the product
        of the gradients of the P1 basis functions i and j.
        """
        return self._stiffness_matrix

    @property
    def stiffness_matrix_solver(self) -> scipy.sparse.linalg.SuperLU:
        """SuperLU solver associated with the free part of the stiffness matrix."""
        return self._stiffness_matrix_solver

    @property
    def mass_matrix(self) -> scipy.sparse.csr_matrix:
        """Mass matrix of the mesh.

        This csr matrix represents a matrix of shape (N_v, N_v) that calculates
        the mass, where the (i,j) component is the integral of the product of
        the P1 basis functions i and j.
        """
        return self._mass_matrix

class SimplyConnectedFreestandingMesh(SingleBodyMesh):
    """2D FEM mesh for simply connected freestanding geometries.

    A 2D FEM mesh catered to simply connected (hole-free) geometries that represent
    superconducting or magnetic microstructures not connected to any source or sink
    outside the field of view.

    Pre-defines basis functions, gradients and matrices such that FEM problems
    and calculations can easily be set up.

    Because there are no sources or sinks, the mesh does not have any boundary
    markers.

    Parameters
    ----------
    vertices : list[list[float]]
        List of 2D coordinates. These are the outer points of the geometry.
    segments: list[list[int]]
        List of vertex index pairs defining which vertices are connected.
    max_area: float
        Maximum area of a single mesh cell, enforced in the triangulation.
        Defaults to 0.1
    min_angle: float
        Minimum angle of a single mesh cell, enforced in the triangulation.
        Defaults to 30.0 degrees.
    verbose: bool
        If true, debug messages are printed to console. Defaults to False.
    refine_distance: float or None
        If not none, the outer boundary gets subdivided until the smallest
        boundary segement has length refine_distance. Essential if a fine
        meshing at the edge, e.g. for London problems, is desired.

    Notes
    -----
    This class differs from SimplyConnectedCurrentMesh only in its use of boundary
    markers, and for sottools v1.0.0 we fix the following TODO: inherit both from a
    new SimplyConnectedMesh base class.
    """

    def __init__(
        self,
        vertices: list[list[float]],
        segments: list[list[int]],
        max_area: float = 0.1,
        min_angle: float = 30.0,
        verbose: bool = False,
        refine_distance: float | None = None,
    ):
        if verbose:
            print("Triangulating mesh...")
            start_time = time.time()

        # Refine
        if refine_distance is not None:
            fine_mesh_data = self._refine_boundary(
                vertices, segments, refine_distance
            )
        else:
            fine_mesh_data = (
                vertices,
                segments,
            )
        vertices_fine, segments_fine = fine_mesh_data

        meshdict = triangle.triangulate(
            {
                "vertices": vertices_fine,
                "segments": segments_fine
            },
            f"pq{min_angle}a{max_area}n",
        )

        if verbose:
            print(f"Triangulation completed in {time.time() - start_time:.2f} seconds")
            print("Next step: calculating matrices...")
            start_time = time.time()

        self._vertices: np.ndarray = meshdict["vertices"]
        self._triangles: np.ndarray = meshdict["triangles"]
        self._neighbors: np.ndarray = meshdict["neighbors"]
        self._segments: np.ndarray = meshdict["segments"]
        self._boundary_vertices = np.array(vertices)

        # Cell areas
        self._areas = self._cell_areas()
        assert (self._areas > 0).all(), "CW triangle(s) found"

        # Cell centers
        self._centers = self._cell_centers()

        # Cell gradients per vertex
        self._shapegradients = self._computeshapegradients()
        assert np.allclose(self._shapegradients.sum(axis=1), 0)

        # Find masks
        self._dirichlet_mask = self._dirichletmask()
        self._free_mask = ~self._dirichlet_mask
        self._free_idx = np.where(self._free_mask)[0]
        self._dirichlet_idx = np.where(self._dirichlet_mask)[0]

        # Cache matrices and operators
        self._edges, self._interior_edges, self._boundary_edges, \
            self._interior_edge_mask, self._boundary_edge_mask, \
            self._internaledgelengths, self._edgedifference_operator = self._compute_edges()

        self._edge_vectors = self._vertices[self._edges[:, 1]] - self._vertices[self._edges[:, 0]]
        self._edge_midpoints = (self._vertices[self._edges[:, 0]] + self._vertices[self._edges[:, 1]]) / 2
        self._edge_lengths = np.linalg.norm(self._edge_vectors, axis=1)

        self._gradient_operator_x, self._gradient_operator_y = self._gradientoperators(
            self._shapegradients
        )
        self._stiffness_matrix = self._stiffnessmatrix(self._shapegradients)
        self._stiffness_matrix_solver = self._freesolver(
            self.stiffness_matrix, self.free_idx
        )  # Caches the (N_free, N_free) part of this matrix for quick solving
        self._mass_matrix = self._massmatrix()

        self._barycentric_areas = self._compute_barycentric_areas() # (N_v,)
        self._cotangent_weights = self._compute_cotangent_weights() # (N_e, )

        self._triangle_edges, self._triangle_edge_signs = self._compute_triangle_edges() # (N_t, 3), with edge k opposing vertex k
        sign_sum = np.bincount(self._triangle_edges.ravel(), weights=self._triangle_edge_signs.ravel(),
                               minlength=len(self.edges))
        assert np.all(sign_sum[self._interior_edge_mask] == 0)
        assert np.all(np.abs(sign_sum[self._boundary_edge_mask]) == 1)

        # The Whitney operators map from edge-based fields to triangle-based fields.
        self._whitney_operator_x, self._whitney_operator_y = self._compute_whitney_operators() # (N_e, N_t)


        N_v = self.vertices.shape[0]
        N_t = self.triangles.shape[0]
        N_e = self.edges.shape[0]
        assert N_e == N_v + N_t - 1, "Euler's formula for planar graphs not satisfied"
        assert self._cotangent_weights[self._interior_edge_mask].min() > -1e-10 * self._cotangent_weights.max(), \
            "Negative interior cotangent weight"

        if verbose:

            print(f"Matrices calculated in {time.time() - start_time:.2f} seconds")
            print("Number of vertices:", N_v)
            print("Number of triangles:", N_t)

    def __getstate__(self) -> dict[str, object]:
        """Conversion of object to dictionary.

        Special care must be taken here because SuperLU objects cannot be
        converted to a dictionary, and can then not be pickled.

        Returns
        -------
        state: dict
            Dictionary representation of the object.
        """
        state = self.__dict__.copy()
        del state["_stiffness_matrix_solver"]  # SuperLU can't be pickled
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Conversion of dictionary to object.

        Special care must be taken here because SuperLU objects cannot be
        converted to a dictionary, and can then not be pickled.

        Parameters
        ----------
        state: dict
            Dictionary representation of the object.
        """
        self.__dict__.update(state)
        # Recompute the factorization
        self._stiffness_matrix_solver = self._freesolver(
            self.stiffness_matrix, self.free_idx
        )

    @staticmethod
    def _refine_boundary(
        vertices: list[list[float]],
        segments: list[list[int]],
        edge_spacing: float = 0.05,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Refines outer boundary of the mesh.

        Parameters
        ----------
        vertices: list[list[float]]
            List of 2D coordinates. These are the outer points of the geometry.
        segments: list[list[int]]
            List of vertex index pairs defining which vertices are connected.
        edge_spacing: float
            Desired spacing between vertices on the boundary. Defaults to 0.05.
        """
        new_vertices = list(vertices)
        new_segments = []

        for seg in segments:
            p0 = np.array(vertices[seg[0]], dtype=float)
            p1 = np.array(vertices[seg[1]], dtype=float)
            length = np.linalg.norm(p1 - p0)
            n_subdivisions = max(1, int(np.ceil(length / edge_spacing)))

            idx_start = seg[0]
            for i in range(1, n_subdivisions):
                t = i / n_subdivisions
                new_point = (1 - t) * p0 + t * p1
                idx_new = len(new_vertices)
                new_vertices.append(new_point.tolist())
                new_segments.append([idx_start, idx_new])
                idx_start = idx_new

            new_segments.append([idx_start, seg[1]])

        return new_vertices, new_segments

    def _cell_areas(self) -> np.ndarray:
        """Signed areas: positive for counter-clockwise oriented triangles."""
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        bs = v1s - v0s
        cs = v2s - v0s
        areas: np.ndarray = 0.5 * (bs[:, 0] * cs[:, 1] - bs[:, 1] * cs[:, 0])

        return areas

    def _cell_centers(self) -> np.ndarray:
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        centers: np.ndarray = (v0s + v1s + v2s) / 3
        return centers

    def _computeshapegradients(self) -> np.ndarray:
        v0s = self.vertices[self.triangles[:, 0]]
        v1s = self.vertices[self.triangles[:, 1]]
        v2s = self.vertices[self.triangles[:, 2]]

        b = np.column_stack(
            [v1s[:, 1] - v2s[:, 1], v2s[:, 1] - v0s[:, 1], v0s[:, 1] - v1s[:, 1]]
        )  # (N_t, 3)
        c = np.column_stack(
            [v2s[:, 0] - v1s[:, 0], v0s[:, 0] - v2s[:, 0], v1s[:, 0] - v0s[:, 0]]
        )  # (N_t, 3)

        shapegrads: np.ndarray = (
            np.stack([b, c], axis=-1) / (2 * self._areas)[:, None, None]
        )
        return shapegrads

    def _dirichletmask(self) -> np.ndarray:
        N_v = self.vertices.shape[0]
        dirichletmask = np.zeros(N_v, dtype=bool)
        verts = np.unique(self._segments[:].ravel())
        dirichletmask[verts] = True
        return dirichletmask

    def _compute_edges(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        scipy.sparse.csr_matrix,
    ]:
        internal_edges = []  # (t1, t2) triangle pairs where t1 and t2 share an edge
        internal_edge_vertices = []  # (v1, v2) vertex pairs for each internal edge
        boundary_edges = []  # (v1, v2) vertex pairs for each boundary edge

        for t, tri in enumerate(self.triangles):
            for i in range(3):
                v1, v2 = tri[(i + 1) % 3], tri[(i + 2) % 3]
                neighbor = self.neighbors[t, i]
                if neighbor != -1:
                    if t < neighbor:  # Avoid double counting
                        internal_edges.append((t, neighbor))
                        internal_edge_vertices.append((v1, v2))
                else:
                    boundary_edges.append((v1, v2))

        (internal_edges_arr, internal_edge_vertices_arr) = (
            np.array(internal_edges),
            np.array(internal_edge_vertices),
        )

        internal_edge_lengths = np.linalg.norm(
            self.vertices[internal_edge_vertices_arr[:, 0]]
            - self.vertices[internal_edge_vertices_arr[:, 1]],
            axis=1,
        )

        N_t = self.triangles.shape[0]
        N_e = len(internal_edges)

        rows = np.arange(N_e)[:, None].repeat(2, axis=1)  # (N_e, 2)
        cols = internal_edges_arr.ravel()  # (N_e, 2)
        edge_difference_operator = scipy.sparse.coo_matrix(
            (np.array([1, -1] * N_e), (rows.ravel(), cols.ravel())), shape=(N_e, N_t)
        ).tocsr()

        all_edges = np.vstack([internal_edge_vertices_arr, boundary_edges])

        # We now have all_edges, internal_edge_vertices, and boundary_edges
        # Both consist of vertex index pairs. We now enforce ordering such that the
        # first index is always smaller than the second.
        for i in range(all_edges.shape[0]):
            if all_edges[i, 0] > all_edges[i, 1]:
                all_edges[i] = all_edges[i][::-1]
        for i in range(internal_edge_vertices_arr.shape[0]):
            if internal_edge_vertices_arr[i, 0] > internal_edge_vertices_arr[i, 1]:
                internal_edge_vertices_arr[i] = internal_edge_vertices_arr[i][::-1]
        for i in range(len(boundary_edges)):
            if boundary_edges[i][0] > boundary_edges[i][1]:
                boundary_edges[i] = boundary_edges[i][::-1]

        # Because all boundary edges come after all internal edges, masking is easy:
        internal_edges_mask = np.zeros(len(all_edges), dtype=bool)
        internal_edges_mask[:len(internal_edge_vertices_arr)] = True
        boundary_edges_mask = ~internal_edges_mask

        return (all_edges,
                internal_edge_vertices_arr,
                np.array(boundary_edges),
                internal_edges_mask,
                boundary_edges_mask,
                internal_edge_lengths,
                edge_difference_operator,
                )

    def _compute_triangle_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the edges of each triangle in the mesh.

        Returns a (N_t, 3) array containing the edge indices of each triangle.
        The edge index corresponds to the edge opposing the vertex with the same index.
        """
        N_t = self.triangles.shape[0]
        N_v = self.vertices.shape[0]
        tri = self.triangles.astype(np.int64)            # int64: keys reach N_v**2
        a = tri[:, [1, 2, 0]]                            # (N_t, 3) start of local edge k
        b = tri[:, [2, 0, 1]]                            # (N_t, 3) end of local edge k

        edge_keys = self.edges[:, 0].astype(np.int64) * N_v + self.edges[:, 1]
        order = np.argsort(edge_keys)
        local_keys = np.minimum(a, b) * N_v + np.maximum(a, b)
        triangle_edges = order[np.searchsorted(edge_keys[order], local_keys)]
        triangle_edge_signs = np.where(a < b, 1, -1)

        return triangle_edges, triangle_edge_signs

    def _compute_whitney_operators(self) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
        N_t = self.triangles.shape[0]
        N_e = self.edges.shape[0]
        grads = self._shapegradients  # (N_t, 3, 2)
        lengths = self._edge_lengths[self._triangle_edges]  # (N_t, 3)
        C = (lengths * self._triangle_edge_signs)[:, :, None] / 3 \
            * (grads[:, [2, 0, 1]] - grads[:, [1, 2, 0]])  # (N_t, 3, 2)

        rows = np.arange(N_t)[:, None].repeat(3, axis=1)  # (N_t, 3)
        cols = self._triangle_edges  # (N_t, 3)
        whitney_x = scipy.sparse.coo_matrix(
            (C[:, :, 0].ravel(), (rows.ravel(), cols.ravel())), shape=(N_t, N_e)
        ).tocsr()
        whitney_y = scipy.sparse.coo_matrix(
            (C[:, :, 1].ravel(), (rows.ravel(), cols.ravel())), shape=(N_t, N_e)
        ).tocsr()
        return whitney_x, whitney_y

    def _gradientoperators(
        self, shapegradients: np.ndarray
    ) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
        N_v = self.vertices.shape[0]
        N_t = self.triangles.shape[0]

        tri_idx = np.arange(N_t)[:, None].repeat(3, axis=1)  # (N_t, 3)
        ver_idx = self.triangles  # (N_t, 3)

        gradient_operator_x = scipy.sparse.coo_matrix(
            (shapegradients[:, :, 0].ravel(), (tri_idx.ravel(), ver_idx.ravel())),
            shape=(N_t, N_v),
        ).tocsr()
        gradient_operator_y = scipy.sparse.coo_matrix(
            (shapegradients[:, :, 1].ravel(), (tri_idx.ravel(), ver_idx.ravel())),
            shape=(N_t, N_v),
        ).tocsr()

        return gradient_operator_x, gradient_operator_y

    def _stiffnessmatrix(self, shapegradients: np.ndarray) -> scipy.sparse.csr_matrix:
        N_v = self.vertices.shape[0]

        K_t = np.einsum("tiu,tju->tij", shapegradients, shapegradients)  # (N_t, 3, 3)
        K_t *= self._areas[:, None, None]  # Scale by area
        rows = self.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        cols = self.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        K = scipy.sparse.coo_matrix(
            (K_t.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()

        return K

    def _freesolver(
        self, matrix: scipy.sparse.csr_matrix, free_idx: np.ndarray
    ) -> scipy.sparse.linalg.SuperLU:
        matrix_free = matrix[free_idx][:, free_idx]
        solver = scipy.sparse.linalg.splu(matrix_free.tocsc())
        return solver

    def _massmatrix(self) -> scipy.sparse.csr_matrix:
        N_v = self.vertices.shape[0]

        local_M = (
            np.where(
                np.eye(3, dtype=bool)[None, :, :],  # (1,3,3)
                1 / 6,  # diagonal entries
                1 / 12,  # off-diagonal entries
            )
            * self._areas[:, None, None]
        )  # (N_t, 3, 3)

        rows = self.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        cols = self.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        M = scipy.sparse.coo_matrix(
            (local_M.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()

        return M

    def _compute_barycentric_areas(self) -> np.ndarray:
        """Compute the barycentric areas for each vertex in the mesh.

        Returns a (N_v,) array containing the barycentric areas of the vertices.
        """
        N_v = self.vertices.shape[0]
        barycentric_areas: np.ndarray = np.bincount(
            self.triangles.ravel(),                 # (3 N_t,) vertex index of each triangle corner
            weights=np.repeat(self.areas / 3, 3),   # (3 N_t,) one third of that triangle's area
            minlength=N_v,
        )
        return barycentric_areas

    def _compute_cotangent_weights(self) -> np.ndarray:
        """Computes the cotangent weight w_ij for all edges e_ij.

        We sort all edges inside self._edges where each entry is an edge e_ij with
        i < j. The cotangent weight w_ij belonging to edge e_ij is simply -1x the
        value of the stiffness matrix K_ij. By sorting this into an array of
        (N_e,) we can easily write vectorize sums over edges."""
        K_edges = self.stiffness_matrix[self.edges[:, 0], self.edges[:, 1]]  # (1, N_e)
        cotangent_weights: np.ndarray = -np.asarray(K_edges).ravel()
        return cotangent_weights

    def field_curl(
        self,
        field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the curl of a field defined on the vertices.

        Represents the operation v = nabla x (f zhat), where the field f
        is a scalar field defined on the vertices. This is exactly how
        streamfunctions are related to sheet current densities in 2D.

        Parameters
        ----------
        field: np.ndarray
            Field defined on the vertices of the mesh. Shape (N_v,).

        Returns
        -------
        Kx: np.ndarray
            x-component of the curl of the field, defined on the cell
            centers. Shape (N_t,).
        Ky: np.ndarray
            y-component of the curl of the field, defined on the cell
            centers. Shape (N_t,).
        """
        field_ts = field[self.triangles]
        grad_field = (field_ts[:, :, None] * self._shapegradients).sum(axis=1)
        Kx = grad_field[:, 1]
        Ky = -grad_field[:, 0]
        return Kx, Ky

    @property
    def vertices(self) -> np.ndarray:
        """Vertices of the mesh.

        Returns a (N_v,2) array of vertex coordinates.
        """
        return self._vertices

    @property
    def triangles(self) -> np.ndarray:
        """Triangulation of the mesh.

        Returns a (N_t,3) array containing the vertex indices of a triangle.
        """
        return self._triangles

    @property
    def areas(self) -> np.ndarray:
        """Areas of the triangles in the mesh.

        Returns a (N_t,) array containing the areas of the triangles.
        """
        return self._areas

    @property
    def centers(self) -> np.ndarray:
        """Centers of the triangles in the mesh.

        Returns a (N_t,2) array containing the coordinates of the triangle centers.
        """
        return self._centers

    @property
    def neighbors(self) -> np.ndarray:
        """Neighbor relationships between triangles.

        Returns a (N_t,3) array containing the triangle indices of the
        neighbors of a triangle.
        """
        return self._neighbors

    @property
    def boundary_vertices(self) -> np.ndarray:
        """Vertices on the boundary of the mesh.

        Returns a (N_b,2) array containing the coordinates of the boundary vertices.
        """
        return self._boundary_vertices

    @property
    def boundary_segments(self) -> np.ndarray:
        """Boundary segments of the mesh.

        Returns a (N_s, 2) array of vertex index pairs defining boundary edges.
        """
        return self._segments

    @property
    def shapegradients(self) -> np.ndarray:
        """Gradients of the basis functions.

        Returns a (N_t,3,2) array containing the gradients of the P1
        linear basis functions for each triangle.
        """
        return self._shapegradients

    @property
    def dirichlet_mask(self) -> np.ndarray:
        """Mask for all Dirichlet boundary conditions on the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the boundary.
        """
        return self._dirichlet_mask

    @property
    def free_mask(self) -> np.ndarray:
        """Mask for free vertices (not on Dirichlet boundaries) of the mesh.

        Returns a boolean array of length N_v, where True indicates a vertex
        is part of the free vertices.
        """
        return self._free_mask

    @property
    def free_idx(self) -> np.ndarray:
        """Indices of free vertices (not on Dirichlet boundaries) of the mesh."""
        return self._free_idx

    @property
    def dirichlet_idx(self) -> np.ndarray:
        """Indices of Dirichlet vertices (on Dirichlet boundaries) of the mesh."""
        return self._dirichlet_idx

    @property
    def internaledgelengths(self) -> np.ndarray:
        """Lengths of the internal edges of the mesh.

        Returns a (N_e,) array containing the lengths of the internal edges of
        the mesh.
        """
        return self._internaledgelengths

    @property
    def edgedifference_operator(self) -> scipy.sparse.csr_matrix:
        """Operator for the difference of a field across internal.

        This csr matrix represents a matrix of shape (N_e, N_t) that calculates
        the difference of a field across internal edges.
        """
        return self._edgedifference_operator

    @property
    def gradient_operator_x(self) -> scipy.sparse.csr_matrix:
        """Operator for the x-gradient of a field.

        This csr matrix represents a matrix of shape (N_t, N_v) that calculates
        the x-gradient of a field defined on the vertices.
        """
        return self._gradient_operator_x

    @property
    def gradient_operator_y(self) -> scipy.sparse.csr_matrix:
        """Operator for the y-gradient of a field.

        This csr matrix represents a matrix of shape (N_t, N_v) that calculates
        the y-gradient of a field defined on the vertices.
        """
        return self._gradient_operator_y

    @property
    def stiffness_matrix(self) -> scipy.sparse.csr_matrix:
        """Stiffness matrix of the mesh.

        This csr matrix represents a matrix of shape (N_v, N_v) that calculates
        the stiffness, where the (i,j) component is the integral of the product
        of the gradients of the P1 basis functions i and j.
        """
        return self._stiffness_matrix

    @property
    def stiffness_matrix_solver(self) -> scipy.sparse.linalg.SuperLU:
        """SuperLU solver associated with the free part of the stiffness matrix."""
        return self._stiffness_matrix_solver

    @property
    def mass_matrix(self) -> scipy.sparse.csr_matrix:
        """Mass matrix of the mesh.

        This csr matrix represents a matrix of shape (N_v, N_v) that calculates
        the mass, where the (i,j) component is the integral of the product of
        the P1 basis functions i and j.
        """
        return self._mass_matrix

    @property
    def edges(self) -> np.ndarray:
        """Edges of the mesh.

        Returns a (N_e, 2) array of vertex index pairs defining edges.
        """
        return self._edges

    @property
    def interior_edges(self) -> np.ndarray:
        """Interior edges of the mesh.

        Returns a (N_ei, 2) array of vertex index pairs defining interior edges.
        """
        return self._interior_edges

    @property
    def boundary_edges(self) -> np.ndarray:
        """Boundary edges of the mesh.

        Returns a (N_eb, 2) array of vertex index pairs defining boundary edges.
        """
        return self._boundary_edges

    @property
    def interior_edge_mask(self) -> np.ndarray:
        """Mask for interior edges of the mesh.

        Returns a boolean array of length N_e, where True indicates an edge
        is an interior edge.
        """
        return self._interior_edge_mask

    @property
    def boundary_edge_mask(self) -> np.ndarray:
        """Mask for boundary edges of the mesh.

        Returns a boolean array of length N_e, where True indicates an edge
        is a boundary edge.
        """
        return self._boundary_edge_mask

    @property
    def edge_vectors(self) -> np.ndarray:
        """Vectors of the edges of the mesh.

        Returns a (N_e, 2) array of vectors corresponding to the edges.
        """
        return self._edge_vectors

    @property
    def edge_midpoints(self) -> np.ndarray:
        """Midpoints of the edges of the mesh.

        Returns a (N_e, 2) array of midpoints corresponding to the edges.
        """
        return self._edge_midpoints

    @property
    def edge_lengths(self) -> np.ndarray:
        """Lengths of the edges of the mesh.

        Returns a (N_e,) array of lengths corresponding to the edges.
        """
        return self._edge_lengths

    @property
    def barycentric_areas(self) -> np.ndarray:
        """Barycentric areas of the vertices of the mesh.

        Returns a (N_v,) array of barycentric areas corresponding to the vertices.
        """
        return self._barycentric_areas

    @property
    def cotangent_weights(self) -> np.ndarray:
        """Cotangent weights of the edges of the mesh.

        Returns a (N_e,) array of cotangent weights corresponding to the edges.
        """
        return self._cotangent_weights

    @property
    def triangle_edges(self) -> np.ndarray:
        """Edges of the triangles of the mesh.

        Returns a (N_t, 3) array of edge indices corresponding to the triangles.
        The edge index corresponds to the edge opposing the vertex with the same index.
        """
        return self._triangle_edges

    @property
    def triangle_edge_signs(self) -> np.ndarray:
        """Signs of the edges of the triangles of the mesh.

        Returns a (N_t, 3) array of signs corresponding to the triangles.
        The sign is +1 if the edge is oriented from the first vertex to the second vertex,
        and -1 if it is oriented from the second vertex to the first vertex.
        The sign corresponds to the edge opposing the vertex with the same index.
        """
        return self._triangle_edge_signs

    @property
    def whitney_operator_x(self) -> scipy.sparse.csr_matrix:
        """Whitney operator for the x-component of the edges of the triangles of the mesh.

        Returns a (N_t, N_e) csr matrix that maps edge-based fields to triangle-based fields.
        """
        return self._whitney_operator_x

    @property
    def whitney_operator_y(self) -> scipy.sparse.csr_matrix:
        """Whitney operator for the y-component of the edges of the triangles of the mesh.

        Returns a (N_t, N_e) csr matrix that maps edge-based fields to triangle-based fields.
        """
        return self._whitney_operator_y


class LondonSolver:
    """Solver for the London equation in a simply connected geometry.

    The solver iteratively solved for the streamfunction and the out-of-
    plane magnetic field until convergence, using dampened Picard
    iteration.

    Parameters
    ----------
    mesh: SimplyConnectedCurrentMesh
        The mesh on which to solve the London equation.
    Pearllength: float
        The Pearl length of the superconductor, in the same units as the mesh.
    B_ext: float
        The external magnetic field applied to the superconductor, in the same
        units as the mesh. The field is uniform and out-of-plane. Defaults to 0.0.
    verbose: bool
        If true, debug messages are printed to console. Defaults to False.

    Attributes
    ----------
    MU0: float
        The magnetic permeability of free space. Default value is 10^-1*4pi,
        which is appropriate for units of um, uT and uA.
    """

    MU0 = 0.1 * 4 * np.pi  # Mu0 for units of um, uT, uA

    def __init__(
        self,
        mesh: SimplyConnectedCurrentMesh,
        Pearllength: float,
        B_ext: float = 0.0,
        verbose: bool = False,
    ):
        self.mesh = mesh
        self.Pearllength = Pearllength
        self.B_ext = B_ext
        self.verbose = verbose

        # Precompute matrices Cx and Cy
        # These link the sheet currents Kx and Ky to the out-of-plane field Bz
        # at the vertices through Bz = (mu0/4pi) * (Cx @ Kx - Cy @ Ky)
        # Dense matrices to calculate from cell-centered sheet current density
        # to vertex-centered z-field
        dx = self.mesh.vertices[:, None, 0] - self.mesh.centers[None, :, 0]
        dy = self.mesh.vertices[:, None, 1] - self.mesh.centers[None, :, 1]
        R3 = (dx**2 + dy**2) ** (3 / 2)
        A = self.mesh.areas[None, :]
        self.Cx = A * dy / R3  # Kx contribution to Bz is Cx*Jy with prefactor
        self.Cy = A * dx / R3  # Ky contribution to Bz is Cy*Jx with prefactor

    def solve(
        self,
        left_bc: float,
        right_bc: float,
        max_iter: int = 50,
        alpha: float = 0.2,
        rtol: float = 1e-5,
        diagnostic_plotter: typing.Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray], None
        ]
        | None = None,
    ) -> dict[str, object]:
        """Solve the London equation for a given mesh and Pearl length.

        Parameters
        ----------
        left_bc: float
            Dirichlet boundary condition for the streamfunction on the left
            side of the mesh. The difference right_bc - left_bc is the total
            current flowing through the mesh.
        right_bc: float
            Dirichlet boundary condition for the streamfunction on the right
            side of the mesh. The difference right_bc - left_bc is the total
            current flowing through the mesh.
        max_iter: int
            Maximum number of iterations for the Picard iteration. Defaults to
            50.
        alpha: float
            Damping factor for the Picard iteration. Defaults to 0.2.
        rtol: float
            Relative tolerance for convergence of the Picard iteration.
            Defaults to 1e-5.
        diagnostic_plotter: callable[[np.ndarray, np.ndarray, np.ndarray,
        np.ndarray], None]
            Optional function to plot the streamfunction, out-of-plane field,
            and sheet current density at each iteration. The function should
            take four arguments: streamfunction, out-of-plane field,
            x-component of sheet current density, and y-component of sheet
            current density.

        Returns
        -------
        Dictionary containing the following keys:
        - "converged": bool, whether the solver converged within max_iter
        - "streamfunction": np.ndarray, the streamfunction at the vertices
        - "Bz": np.ndarray, the out-of-plane magnetic field at the vertices
        - "Kx": np.ndarray, the x-component of the sheet current density at the
        cell centers
        - "Ky": np.ndarray, the y-component of the sheet current density at the
        cell centers
        """
        # Initial streamfunction is just the solution without magnetic feedback
        Bz_initial = np.ones(self.mesh.vertices.shape[0]) * self.B_ext
        g = self._solve_streamfunction(left_bc, right_bc, Bz=Bz_initial)

        # Initial field is the response to this streamfunction
        Kx, Ky = self.mesh.field_curl(g)
        Bz = self._calculate_Bz(Kx, Ky) + self.B_ext

        # Iteratively solve for streamfunction and field until convergence
        success = False
        for i in range(max_iter):
            if self.verbose:
                t_start = time.time()
            g_new = self._solve_streamfunction(left_bc, right_bc, Bz)
            g_old = g.copy()
            g = g * (1 - alpha) + g_new * alpha
            Kx, Ky = self.mesh.field_curl(g)
            Bz = self._calculate_Bz(Kx, Ky) + self.B_ext
            if self.verbose:
                print(
                    f"Iteration {i + 1} completed in "
                    f"{time.time() - t_start:.2f} seconds"
                )
                t_start = time.time()

            # Check for convergence
            rdiff = np.linalg.norm(g_new - g_old) / np.linalg.norm(g)

            if self.verbose:
                print(
                    f"Iter {i}: max|Δg|={np.max(np.abs(g - g_new)):.2e}, "
                    f"max|Bz|={np.max(np.abs(Bz)):.2e}, "
                    f"nan in Bz: {np.isnan(Bz).any()}, inf in Bz: "
                    f"{np.isinf(Bz).any()}, "
                    f"relative change in g: {rdiff:.2e}"
                )

            if diagnostic_plotter is not None:
                diagnostic_plotter(g, Bz, Kx, Ky)

            if rdiff < rtol:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                success = True
                break

        return {
            "converged": success,
            "streamfunction": g if success else None,
            "Bz": Bz if success else None,
            "Kx": Kx if success else None,
            "Ky": Ky if success else None,
        }

    def _solve_streamfunction(
        self, left_bc: float, right_bc: float, Bz: np.ndarray
    ) -> np.ndarray:
        magnetic_prefactor = 2 / (self.MU0 * self.Pearllength)

        N_v = self.mesh.vertices.shape[0]

        g_prescribed = np.zeros(N_v)
        g_prescribed[self.mesh.dirichletleft_mask] = left_bc
        g_prescribed[self.mesh.dirichletright_mask] = right_bc
        g_D = g_prescribed[self.mesh.dirichlet_idx]

        rhs_boundary = (
            -self.mesh.stiffness_matrix[self.mesh.free_idx][:, self.mesh.dirichlet_idx]
            @ g_D
        )

        rhs_source = -magnetic_prefactor * (
            self.mesh.mass_matrix[self.mesh.free_idx] @ Bz
        )

        rhs = rhs_boundary + rhs_source

        g_free = self.mesh.stiffness_matrix_solver.solve(rhs)
        g = np.zeros(N_v)
        g[self.mesh.free_idx] = g_free
        g[self.mesh.dirichlet_idx] = g_D
        return g

    def _calculate_Bz(self, Kx: np.ndarray, Ky: np.ndarray) -> np.ndarray:
        Bz: np.ndarray = (self.MU0 / (4 * np.pi)) * (self.Cx @ Kx - self.Cy @ Ky)
        return Bz


class HydroSolver:
    """Solver for the electron-hydrodynamic equations in a simply connected geometry.

    Solves the electron-hydrodynamic equations (EHE) in the streamfunction-vorticity
    formulation:

    -tau/(e*n) (dg/dy domega/dx - dg/dx domega/dy) - D^2 nabla^2 omega + omega = 0
    nabla^2 g = -omega

    We use the stiffness and mass matrices K and M of the mesh to discretize the
    equations. The boundary conditions are no-slip and dirichlet on the vacuum
    boundaries for only the streamfunction (i.e. we fix the total current). K and M, as
    constructed, are of shape (N_v, N_v) but we often need (N_free, N_v), (N_free,
    N_free), etc. For this we use shorthand K^F, K^FF, etc. In this way, the EHE reduce
    to:

    K^FF g_F - M^F omega = -K^FD g_D
    (-tau/e*n V_F + D^2 K^F + M^F) omega = 0
    -K^DF g^F + M^D omega = K^DD g^D

    Which can be iteratively solved, but only as a block system. The V matrix has to be
    constructed from a lagging streamfunction g and is constructed from g and the
    C-tensor of the mesh, where the C-tensor is

    C_ijk = int dA (nabla phi_j) x (nabla phi_k) . zhat phi_i

    V_ij is then sum_k C_ijk g_k. g converges through Picard iteration.

    Parameters
    ----------
    mesh: SimplyConnectedCurrentMesh
        The mesh on which to solve the EHE.
    ten: float
        The parameter tau / e n, where tau is the momentum relaxation time, e is the
        electron charge, and n is the electron density. Defaults to 1.0.
    D: float
        The viscous length. Defaults to 1.0.
    verbose: bool, optional
        If true, debug messages are printed to console. Defaults to False.
    """

    def __init__(
        self,
        mesh: SimplyConnectedCurrentMesh,
        ten: float = 1.0,
        D: float = 1.0,
        verbose: bool = False,
    ) -> None:
        self.mesh = mesh
        self.ten = ten  # tau / e n
        self.D = D  # diffusion constant
        self.verbose = verbose

    def _v_matrix(self, g: np.ndarray) -> scipy.sparse.csr_matrix:
        """Construct the V matrix for the hydrodynamic solver.

        Parameters
        ----------
        g: np.ndarray
            The streamfunction values at the vertices. Shape (N_v,).

        Returns
        -------
        V_F: scipy.sparse.csr_matrix
            The V matrix restricted to the free vertices. Shape (N_free, N_v).
        """
        N_t = self.mesh.triangles.shape[0]
        N_v = self.mesh.vertices.shape[0]
        g_t = g[self.mesh.triangles]  # (N_t, 3)
        weights = (np.roll(g_t, -1, axis=1) - np.roll(g_t, -2, axis=1)) / 6
        V_entries = np.broadcast_to(weights[:, None, :], (N_t, 3, 3))
        cols = self.mesh.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        rows = self.mesh.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        V = scipy.sparse.coo_matrix(
            (V_entries.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()
        V_F = V[self.mesh.free_idx]  # (N_free, N_v)
        return V_F

    def _step(
        self,
        left_bc: float,
        right_bc: float,
        g_old: np.ndarray,  # (N_v,)
    ) -> tuple[np.ndarray, np.ndarray]:
        # Solve the hydrodynamic equations for g and omega
        # in block format. The solution is a single vector of length (N_free + N_v,)
        # where the N_free part corresponds to the free vertices of g and the N_v part
        # corresponds to omega.

        # Equations:
        # K^FF g_F - M^F omega = -K^FD g_D
        # (-ten V_F + D^2 K^F + M^F) omega = 0
        # -K^DF g^F + M^D omega = K^DD g^D

        V_F = self._v_matrix(g_old)  # (N_free, N_v)
        K_F = self.mesh.stiffness_matrix[self.mesh.free_idx]  # (N_free, N_v)
        K_FF = self.mesh.stiffness_matrix[self.mesh.free_idx][:, self.mesh.free_idx]
        K_FD = self.mesh.stiffness_matrix[self.mesh.free_idx][
            :, self.mesh.dirichlet_idx
        ]
        K_DF = self.mesh.stiffness_matrix[self.mesh.dirichlet_idx][
            :, self.mesh.free_idx
        ]
        K_DD = self.mesh.stiffness_matrix[self.mesh.dirichlet_idx][
            :, self.mesh.dirichlet_idx
        ]
        M_F = self.mesh.mass_matrix[self.mesh.free_idx]
        M_D = self.mesh.mass_matrix[self.mesh.dirichlet_idx]

        block_matrix = scipy.sparse.bmat(
            [
                [K_FF, -M_F],
                [None, -self.ten * V_F + self.D**2 * K_F + M_F],
                [-K_DF, M_D],
            ],
            format="csc",
        )

        g = np.zeros(self.mesh.vertices.shape[0])
        g[self.mesh.dirichletleft_mask] = left_bc
        g[self.mesh.dirichletright_mask] = right_bc
        g_D = g[self.mesh.dirichlet_idx]
        rhs_1 = -K_FD @ g_D
        rhs_2 = np.zeros(self.mesh.free_idx.shape[0])
        rhs_3 = K_DD @ g_D

        rhs = np.concatenate([rhs_1, rhs_2, rhs_3])

        solution = scipy.sparse.linalg.spsolve(block_matrix, rhs)

        g[self.mesh.free_idx] = solution[: self.mesh.free_idx.shape[0]]
        omega = solution[self.mesh.free_idx.shape[0] :]
        return g, omega

    def solve(
        self,
        left_bc: float,
        right_bc: float,
        alpha: float = 0.8,
        max_iter: int = 50,
        rtol: float = 1e-5,
    ) -> dict[str, object]:
        """Solve the hydrodynamic equations for a given mesh and parameters.

        Parameters
        ----------
        left_bc: float
            The Dirichlet boundary condition on the left boundary.
        right_bc: float
            The Dirichlet boundary condition on the right boundary.
        alpha: float, optional
            The relaxation parameter for the iterative solver. Defaults to 0.8.
        max_iter: int, optional
            The maximum number of iterations for the solver. Defaults to 50.
        rtol: float, optional
            The relative tolerance for convergence. Defaults to 1e-5.

        Returns
        -------
        dict[str, object]
            A dictionary containing the results of the solver, including:
            - "converged": bool, whether the solver converged
            - "streamfunction": np.ndarray, the streamfunction values
            - "vorticity": np.ndarray, the vorticity values
            - "Kx": np.ndarray, the x-component of the sheet current density
            - "Ky": np.ndarray, the y-component of the sheet current density
        """
        g = np.zeros(self.mesh.vertices.shape[0])
        omega = np.zeros(self.mesh.vertices.shape[0])
        g[self.mesh.dirichletleft_mask] = left_bc
        g[self.mesh.dirichletright_mask] = right_bc

        did_converge = False
        for iter in range(max_iter):
            g_new, omega_new = self._step(left_bc, right_bc, g)

            g = (1 - alpha) * g + alpha * g_new
            omega = (1 - alpha) * omega + alpha * omega_new

            rdiff_g = np.linalg.norm(g_new - g) / np.linalg.norm(g)
            rdiff_omega = np.linalg.norm(omega_new - omega) / np.linalg.norm(omega)

            if self.verbose:
                print(
                    f"Iter {iter}: relative change in g: {rdiff_g:.2e}, "
                    f"relative change in omega: {rdiff_omega:.2e}"
                )

            if rdiff_g < rtol and rdiff_omega < rtol:
                if self.verbose:
                    print(f"Converged after {iter + 1} iterations")
                did_converge = True
                break

        Kx, Ky = self.mesh.field_curl(g)

        res = {
            "converged": did_converge,
            "streamfunction": g,
            "vorticity": omega,
            "Kx": Kx,
            "Ky": Ky,
        }

        return res


class InhomogeneousOhmicSolver:
    """Solver for the inhomgeneous Ohm's law in a simply connected geometry.

    Solves for the current flow in a metal of inhomgeneous resistivity rho(x,y). In the
    streamline formalism, Ohm's law E = rho J then becomes

    rho nabla^2 g + (nabla rho) dot (nabla g) = 0

    The normal stiffness matrix K_ij has elements int dA (nabla phi_i) dot (nabla
    phi_j). In this case, we introduce a new stiffness matrix K_ij^rho with elements
    int dA rho(x,y) (nabla phi_i) dot (nabla phi_j). Approximating rho as constant over
    each triangle t, with area A_t, this matrix becomes tractable.

    In weak form, we then obtain

    K^rho,FF g_F = -K^rho,FD g_D

    Which can be solved through standard methods.

    Parameters
    ----------
    mesh: SimplyConnectedCurrentMesh
        The mesh on which to solve the inhomogeneous Ohm's law.
    rho: np.ndarray
        The resistivity at each triangle. Shape (N_t,).
    verbose: bool, optional
        If true, debug messages are printed to console. Defaults to False.
    """

    def __init__(
        self,
        mesh: SimplyConnectedCurrentMesh,
        rho: np.ndarray,
        verbose: bool = False,
    ) -> None:
        assert rho.shape[0] == mesh.triangles.shape[0], (
            "Resistivity array must have the same length as the number of triangles in "
            "the mesh."
        )

        self.mesh = mesh
        self.rho = rho
        self.verbose = verbose

        # Precompute the resistivity-weighted stiffness matrix
        self.K_rho = self._compute_resistivity_weighted_stiffness_matrix()

    def _compute_resistivity_weighted_stiffness_matrix(self) -> scipy.sparse.csr_matrix:
        """Compute the resistivity-weighted stiffness matrix K^rho.

        Returns
        -------
        K_rho: scipy.sparse.csr_matrix
            The resistivity-weighted stiffness matrix. Shape (N_v, N_v).
        """
        N_v = self.mesh.vertices.shape[0]
        local_K_rho = np.einsum(
            "tiu,tju->tij", self.mesh.shapegradients, self.mesh.shapegradients
        ) * (self.rho[:, None, None] * self.mesh.areas[:, None, None])  # (N_t, 3, 3)

        rows = self.mesh.triangles[:, :, None].repeat(3, axis=2)  # (N_t, 3, 3)
        cols = self.mesh.triangles[:, None, :].repeat(3, axis=1)  # (N_t, 3, 3)
        K_rho = scipy.sparse.coo_matrix(
            (local_K_rho.ravel(), (rows.ravel(), cols.ravel())), shape=(N_v, N_v)
        ).tocsr()

        return K_rho

    def solve(
        self,
        left_bc: float,
        right_bc: float,
    ) -> dict[str, np.ndarray]:
        """Solve the inhomogeneous Ohm's law for a given mesh and resistivity.

        Parameters
        ----------
        left_bc: float
            The Dirichlet boundary condition on the left boundary.
        right_bc: float
            The Dirichlet boundary condition on the right boundary.

        Returns
        -------
        res: dict[str, np.ndarray]
            A dictionary containing the results of the solver, including:
            - "streamfunction": np.ndarray, the streamfunction values
            - "Kx": np.ndarray, the x-component of the sheet current density
            - "Ky": np.ndarray, the y-component of the sheet current density
        """
        N_v = self.mesh.vertices.shape[0]

        g_prescribed = np.zeros(N_v)
        g_prescribed[self.mesh.dirichletleft_mask] = left_bc
        g_prescribed[self.mesh.dirichletright_mask] = right_bc
        g_D = g_prescribed[self.mesh.dirichlet_idx]

        rhs_boundary = -self.K_rho[self.mesh.free_idx][:, self.mesh.dirichlet_idx] @ g_D

        # Solve for free vertices
        K_rho_free = self.K_rho[self.mesh.free_idx][:, self.mesh.free_idx]
        K_rho_free = K_rho_free.tocsc()  # Convert to CSC format for spsolve
        g_free = scipy.sparse.linalg.spsolve(K_rho_free, rhs_boundary)

        # Combine free and Dirichlet values
        g = np.zeros(N_v)
        g[self.mesh.free_idx] = g_free
        g[self.mesh.dirichlet_idx] = g_D

        Kx, Ky = self.mesh.field_curl(g)

        res = {
            "streamfunction": g,
            "Kx": Kx,
            "Ky": Ky,
        }

        return res
