import dataclasses
import numpy as np
import torch
import scipy.sparse
from sottools.mesh import SimplyConnectedFreestandingMesh

DTYPE = torch.float32

@dataclasses.dataclass(frozen=True)
class Superconductor:
    """All parameters that describe a GL superconductor.

    Parameters
    ----------
    coherence_length: float
        The coherence length of the superconductor, in microns. Finite temperature
        effects are not included, so increase the coherence length to whatever value
        holds at the temperature of interest.
    penetration_depth: float
        The penetration depth of the superconductor, in microns. This is the London
        penetration depth, not the Pearl length. Finite temperature effects are not
        included, so increase the penetration depth to whatever value holds at the
        temperature of interest.
    thickness: float
        The thickness of the superconductor, in microns.
    """

    coherence_length: float
    penetration_depth: float
    thickness: float

class GLInverter:
    """Minimize the Ginzburg Landau free energy together with image fidelity.

    The full version of the class solved the Ginzburg-Landau free energy, including
    diamagnetic screening, while simulteaneously calculating the corresponding
    magnetic image. By minimizing a composite loss function of both the image
    fidelity and the free energy, a wavefunction can be reconstructed from a magnetic
    image. Passing no image to the constructor will result in a pure, iterative
    GL solver.

    Unit-wise, we retain the sottools convention of working directly in uA, um, uT
    units. However, to retain numerical precision, we cannot equivalently express the
    wavefunction in natural units. Instead, we normalize the wavefunction to its
    equilibrium value. In this case, the GL free energy density (wavefunction part
    only):

    f = e_0 * {(-|psi|^2 + 0.5 |psi|^4)/xi^2 lambda^2 +
    |(-i nabla-2pi/Phi0 A)psi|^2/lambda^2}

    Where e_0 is the universal constant Phi0^2/(8pi^2 mu0), which has units J m. We
    will express it as e_0 = 4.309544e4 uT uA um^3.

    For the initial conditions, we must start close to zero to ensure vortices can
    easily enter. At the same time, zero is an extremum of the free energy.
    Consequently, we use seeded random noise to initialize the wavefunction in a
    range wf_sigma near zero.

    Parameters
    ----------
    mesh : SimplyConnectedFreestandingMesh
        The mesh on which the GL free energy is defined.
    superconductor : Superconductor
        The parameters of the superconductor.
    image : np.ndarray | None
        The magnetic image to be fitted. If None, the class will only minimize the
        GL free energy.
    B_ext : float, optional
        The external magnetic field applied to the superconductor, in uT. Default
        is zero.
    screening : bool, optional
        Whether to include diamagnetic screening. If True, the simulation will be more
        accurate, but it will also slow down due to the required dense matrix
        operations. Each evaluation of the free energy will require a new calculation
        of the linking variables. If False, no Meissner will be included (the field
        everywhere is the applied field), but because the linking variables can be
        cached, the simulation is much faster. Default is False.
    device : torch.device optional
        The device on which the calculations will be performed. Default is cpu.
    seed : int, optional
        The random seed for the torch random number generator. Default is 0.
    wf_sigma : float, optional
        The standard deviation of the initial wavefunction noise. Default is 0.01.
    """

    def __init__(
        self,
        mesh: SimplyConnectedFreestandingMesh,
        superconductor: Superconductor,
        image: np.ndarray | None = None,
        B_ext: float = 0.0,
        screening: bool = False,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        wf_sigma: float = 0.01,
    ):
        self._mesh = mesh
        self._superconductor = superconductor
        self._B_ext = B_ext
        self._screening = screening
        self._image = image
        self._device = device
        self._seed = seed
        self._wf_sigma = wf_sigma
        self._generator = torch.Generator(device=device).manual_seed(seed)

        self._loss_history = [] # Updated each LBFGS call with the latest loss function,
        # which is a sum of the scaled free energy and the image fidelity.

        # Prepare GL constants
        self._mu0 = 0.1 * 4 * np.pi # uT um / uA
        self._phi0 = 2067.83 # uT um^2
        self._e0 = self._phi0**2 / (8 * np.pi**2 * self._mu0)   # uT uA um^3, approx 4.31e4

        # Cache mesh properties that are needed as torch tensors
        self._barycentric_areas = torch.as_tensor(
            self._mesh.barycentric_areas, dtype=DTYPE, device=device) # (N_v,)
        self._i = torch.from_numpy(self._mesh.edges[:, 0]
        ).to(device=device, dtype=torch.long) # (N_e,), 1st vertex of each edge
        self._j = torch.from_numpy(self._mesh.edges[:, 1]
        ).to(device=device, dtype=torch.long) # (N_e,), 2nd vertex of each edge
        self._w = torch.from_numpy(
            self._mesh.cotangent_weights
        ).to(device=device, dtype=DTYPE) # (N_e,), cotangent weight of each edge
        self._edge_lengths = torch.from_numpy(self._mesh.edge_lengths).to(
            device=device, dtype=DTYPE
        )  # (N_e,), length of each edge
        grads = (
            self._mesh.shapegradients
        )  # (N_t, 3, 2), grad of P1 basis per local vertex

        def to_torch_sparse(M: scipy.sparse.spmatrix) -> torch.Tensor:
            M = M.tocoo()
            return torch.sparse_coo_tensor(
                np.vstack([M.row, M.col]),
                M.data,
                size=M.shape,
                dtype=DTYPE,
                device=device,
            ).coalesce()

        self._whitney_x = to_torch_sparse(mesh.whitney_operator_x)  # (N_t, N_e)
        self._whitney_y = to_torch_sparse(mesh.whitney_operator_y)  # (N_t, N_e)

        # Whether or not screening is defined, we pre-allocate a tensor for the
        # linking variables. There is one complex linking variable for each edge e_ij:
        # U_ij = exp(i A_ij) where A_ij = 2pi/Phi0 * integral(A . dl) along the edge.
        # A can be either a fully external vector potential (no screening) or self
        # consistently determined. We store U_ij as two real numbers, c for
        # cos(A_ij) and s for sin(A_ij).
        # If no screening, we can pre-compute once and never touch them again
        if not self._screening:
            self._c, self._s = self._cache_constant_linking_variables()
        else:
            self._c = torch.zeros(self._mesh.edges.shape[0], dtype=DTYPE, device=device)
            self._s = torch.zeros(self._mesh.edges.shape[0], dtype=DTYPE, device=device)

        # Prepare the inversion variables
        N_v = self._mesh.vertices.shape[0]
        self._u = torch.randn(N_v, dtype=DTYPE, device=device, generator=self._generator) * self._wf_sigma
        self._v = torch.randn(N_v, dtype=DTYPE, device=device, generator=self._generator) * self._wf_sigma
        self._u.requires_grad = True
        self._v.requires_grad = True


    def _free_energy(self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total Ginzburg-Landau free energy for a given wavefunction,

        Parameters
        ----------
        u: torch.Tensor
            The real part of the wavefunction, shape (N_v,)
        v: torch.Tensor
            The imaginary part of the wavefunction, shape (N_v,)

        Returns
        -------
        free_energy: torch.Tensor
            The total Ginzburg-Landau free energy, 0-dimensional tensor
        """
        free_energy = self._potential_energy(u, v) + self._kinetic_energy(u, v)
        return free_energy

    def _potential_energy(self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the potential energy of the Ginzburg-Landau free energy.

        Parameters
        ----------
        u: torch.Tensor
            The real part of the wavefunction, shape (N_v,)
        v: torch.Tensor
            The imaginary part of the wavefunction, shape (N_v,)

        Returns
        -------
        potential_energy: torch.Tensor
            The potential energy of the Ginzburg-Landau free energy, 0-dimensional
            tensor
        """
        psi_squared = u**2 + v**2
        energy_density = -psi_squared + 0.5 * psi_squared**2
        t = self._superconductor.thickness
        lamb = self._superconductor.penetration_depth
        xi = self._superconductor.coherence_length
        prefactor = self._e0 * t / (xi**2 * lamb**2)
        potential_energy = prefactor * torch.sum(energy_density * self._barycentric_areas)
        return potential_energy

    def _kinetic_energy(self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the kinetic energy of the Ginzburg-Landau free energy.

        Parameters
        ----------
        u: torch.Tensor
            The real part of the wavefunction, shape (N_v,)
        v: torch.Tensor
            The imaginary part of the wavefunction, shape (N_v,)

        Returns
        -------
        kinetic_energy: torch.Tensor
            The kinetic energy of the Ginzburg-Landau free energy, 0-dimensional tensor
        """
        # Given updated linking variables Uij = cij + i sij, the kinetic energy is
        # given by a simple sum over all edges (Bishop v Horn)
        # E_kin = sum_{e_ij} w_ij * |U_ij psi_i - psi_j|^2
        if self._screening:
            raise NotImplementedError("Screening is not yet implemented.")
        else:
            c, s = self._c, self._s
        lamb = self._superconductor.penetration_depth
        t = self._superconductor.thickness
        prefactor = self._e0 * t / (lamb**2)
        kinetic_energy = prefactor * torch.sum(
            self._w * (
                (c * u[self._i] - s * v[self._i] - u[self._j])**2 +
                (s * u[self._i] + c * v[self._i] - v[self._j])**2
            )
        )
        return kinetic_energy


    def _cache_constant_linking_variables(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute the linking variables for the case of no screening.

        The linking variables are defined as U_ij = exp(i A_ij) where A_ij is the
        integral of the vector potential along the edge e_ij. If there is no
        screening, then A is simply the external vector potential, which is constant
        and can be pre-computed once and never touched again. A_ext is determined by
        the applied field B_ext. We use a simple gauge: A_ext = B_ext * (-y, 0). The
        linking variable is split into c = cos(A_ij) and s = sin(A_ij).
        """
        midpoints = torch.from_numpy(self._mesh.edge_midpoints).to(dtype=DTYPE, device=self._device) # (N_e, 2)
        eij = torch.from_numpy(self._mesh.edge_vectors).to(dtype=DTYPE, device=self._device) # (N_e, 2), edge vectors
        A_ext = self._B_ext * torch.stack(
            [-midpoints[:, 1], torch.zeros_like(midpoints[:, 0])],
            dim=1
        ) # (N_e, 2)
        Aij = 2 * torch.pi / self._phi0 * torch.sum(A_ext * eij, dim=1) # (N_e,)

        c = torch.cos(Aij) # (N_e,)
        s = torch.sin(Aij) # (N_e,)

        return c, s

    def fit(
            self,
            n_steps: int = 100,
            convergence_atol: float | None = None,
            convergence_rtol: float | None = None,
            lbfgs_maxiter: int = 10,
            lbfgs_gtol: float = 1e-6,
            lbfgs_ftol: float = 1e-6,
    ) -> dict[str, int | bool | str]:
        """Fit the wavefunction to the image.

        If no image is supplied, the GL energy is simply minimized. If an image is
        supplied, the GL energy and the image fidelity are minimized together.

        The internally kept wavefunction is the fitted object. Consequently, the fit
        can always be extended by calling fit again.

        The strategy is to use many quick calls to the LBFGS optimizer, and log loss
        in between. If convergence_atol and convergence_rtol are kept at None, the full
        n_steps are performed. Otherwise, the fit automatically stops when the
        loss difference between two consecutive LBFGS calls is less than the absolute
        or relative tolerance.
        """

        total_area = self._mesh.barycentric_areas.sum()
        t = self._superconductor.thickness
        lamb = self._superconductor.penetration_depth
        xi = self._superconductor.coherence_length
        optimizer_scale = self._e0 * t / lamb**2  # Mesh-independent gradient scale
        log_scale = self._e0 * t * total_area / (xi**2 * lamb**2)  # Uniform state: -0.5

        optimizer = torch.optim.LBFGS(
            [self._u, self._v],
            lr=1,
            max_iter=lbfgs_maxiter,
            max_eval=lbfgs_maxiter * 5,
            tolerance_grad=lbfgs_gtol,
            tolerance_change=lbfgs_ftol,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            free_energy = self._free_energy(self._u, self._v)
            if self._image is not None:
                raise NotImplementedError("Image fidelity is not yet implemented.")
            loss = free_energy / optimizer_scale
            loss.backward()
            return loss

        did_converge = False
        for step in range(n_steps):
            optimizer.step(closure)
            with torch.no_grad():
                loss_after = float(self._free_energy(self._u, self._v) / log_scale)
            self._loss_history.append(float(loss_after))
            if convergence_atol is not None or convergence_rtol is not None:
                if step > 0:
                    loss_prev = self._loss_history[-2]
                    loss_diff = abs(loss_after - loss_prev)
                    if convergence_atol is not None and loss_diff < convergence_atol:
                        did_converge = True
                        break
                    if convergence_rtol is not None and loss_diff / abs(loss_prev) < convergence_rtol:
                        did_converge = True
                        break

        return {
            "steps_total": step + 1,
            "converged": did_converge,
            "message": "converged" if did_converge else "max steps reached",
        }

    def _edge_currents(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes the sheet currents in the edge current basis.

        Each edge e_ij is associated with a current

        K_ij  = K(0.5*(ri+rj)) dot hat(e_ij)

        These N_e elements can be calculated from the wavefunction and the linking
        variables through

        K_ij = Phi0 t / (2pi mu0 lambda^2) * Im[psi_i^* U_ij^* psi_j]/e_ij
        """

        t = self._superconductor.thickness
        lamb = self._superconductor.penetration_depth
        prefactor = self._phi0 * t / (2 * np.pi * self._mu0 * lamb**2)
        c, s = self._c, self._s

        # Im psi_i^* U_ij psi_j = u_i v_j c - u_j v_i c + u_i u_j s + v_i v_j s
        ui = u[self._i]
        uj = u[self._j]
        vi = v[self._i]
        vj = v[self._j]
        Im_psiUpsi = ui * vj * c - uj * vi * c - ui * uj * s - vi * vj * s

        eij = self._edge_lengths

        Kij = prefactor * Im_psiUpsi / eij  # (N_e,)

        return Kij

    def _sheet_currents(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the sheet currents as piecewise constant vectors per triangle."""

        Kij = self._edge_currents(u, v) # (N_e,)
        Kx = torch.sparse.mm(self._whitney_x, Kij.unsqueeze(1)).squeeze(1)  # (N_t,)
        Ky = torch.sparse.mm(self._whitney_y, Kij.unsqueeze(1)).squeeze(1)  # (N_t,)

        return Kx, Ky


    @property
    def wavefunction(self) -> np.ndarray:
        """Returns the wavefunction as a complex numpy array of shape (N_v,)."""
        return self._u.detach().cpu().numpy() + 1j * self._v.detach().cpu().numpy()

