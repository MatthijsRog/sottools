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
        verbose: bool = False,
    ):
        self.mesh = mesh
        self.Pearllength = Pearllength
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
        g = self._solve_streamfunction(
            left_bc, right_bc, Bz=np.zeros(self.mesh.vertices.shape[0])
        )

        # Initial field is the response to this streamfunction
        Kx, Ky = self.mesh.field_curl(g)
        Bz = self._calculate_Bz(Kx, Ky)

        # Iteratively solve for streamfunction and field until convergence
        success = False
        for i in range(max_iter):
            if self.verbose:
                t_start = time.time()
            g_new = self._solve_streamfunction(left_bc, right_bc, Bz)
            g_old = g.copy()
            g = g * (1 - alpha) + g_new * alpha
            Kx, Ky = self.mesh.field_curl(g)
            Bz = self._calculate_Bz(Kx, Ky)
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
