import dataclasses

import numpy as np
import pytest
import torch

import sottools.backward as backward
import sottools.forward as forward
from sottools.mesh import InhomogeneousOhmicSolver, SimplyConnectedCurrentMesh
from sottools.mesh import SimplyConnectedFreestandingMesh

BM = SimplyConnectedCurrentMesh.BoundaryMarker


@pytest.fixture(scope="session")
def bar_mesh() -> SimplyConnectedCurrentMesh:
    """8x3 rectangular bar, current flows in +x."""
    vertices = [[0, 0], [0, 3], [8, 3], [8, 0]]
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
    ]
    return SimplyConnectedCurrentMesh(vertices, segments, segment_markers, max_area=0.1)


@pytest.fixture(scope="session")
def bar_mesh_london() -> SimplyConnectedCurrentMesh:
    """8x2 rectangular bar, current flows in +x."""
    vertices = [[-4, -1], [-4, 1], [4, 1], [4, -1]]
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
    ]
    return SimplyConnectedCurrentMesh(
        vertices, segments, segment_markers, max_area=0.02, refine_distance=0.05
    )


@pytest.fixture(scope="session")
def complex_mesh_input() -> (list[list[float]], list[list[int]], list[BM]):
    # Right-aligned constriction

    vertices = [
        [0.0, 0.0],
        [0.0, 4.0],
        [9.0, 4.0],
        [9.0, 0.0],
        [5.0, 0.0],
        [5.0, 3.0],
        [4.0, 3.0],
        [4.0, 0.0],
    ]

    segments = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]]

    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
        BM.DIRICHLET_LEFT,
    ]

    return vertices, segments, segment_markers


@pytest.fixture(scope="session")
def inverter1d_with_random_currents() -> tuple[backward.Inverter1D, np.ndarray]:
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    params = forward.Forward1DParameters(
        Lscan=10.0, Wdevice=5, Nscan=200, phi=np.deg2rad(phi)
    )

    y = params.ydevice.detach().cpu().numpy()
    Ngaussians = 5
    sigmas = rng.uniform(0.1, 0.5, size=Ngaussians)
    ymeans = rng.uniform(-2.5, 2.5, size=Ngaussians)
    amplitudes = rng.uniform(0.5, 1.5, size=Ngaussians)
    current = np.zeros_like(y)
    for sigma, ymean, amplitude in zip(sigmas, ymeans, amplitudes, strict=True):
        current += amplitude * np.exp(-((y - ymean) ** 2) / (2 * sigma**2))

    fwd = forward.Forward1D(params)
    signal = fwd.forward(current)
    zeroregularizer = 0.0 * backward.Tikhonov1()

    inverter = backward.Inverter1D(fwd, signal, zeroregularizer)

    return inverter, current


def _build_coinv_problem(
    Lscan,
    Nscan,
    Wdevice,
    SNR,
    rho1_true,
    rho2_true,
    h_true,
    phi_true,
    invert_normal=True,
    invert_scan=False,
):
    rng = np.random.default_rng(0)

    params = forward.Forward1DParameters(
        Lscan=Lscan,
        Wdevice=Wdevice,
        Nscan=Nscan,
        rho1=rho1_true,
        rho2=rho2_true,
        height=h_true,
        phi=phi_true,
        invert_normal=invert_normal,
        invert_scan=invert_scan,
    )

    deviceax = np.copy(params.ydevice.numpy())
    pos1 = -1.2
    pos2 = 1.6
    sigma = 0.6
    currents = np.exp(-((deviceax - pos1) ** 2) / (2 * sigma**2)) + np.exp(
        -((deviceax - pos2) ** 2) / (2 * sigma**2)
    )

    forw = forward.Forward1D(params)
    signal = forw.forward(currents)
    noise_sigma = np.ptp(signal) / SNR
    signal_noise = signal + rng.normal(0, noise_sigma, size=signal.shape)

    return forw, signal_noise, currents, params, noise_sigma


@pytest.fixture(scope="session")
def coinv_problem():
    return _build_coinv_problem(
        Lscan=10,
        Nscan=100,
        Wdevice=5,
        SNR=200,
        rho1_true=0.2,
        rho2_true=0.3,
        h_true=0.5,
        phi_true=np.deg2rad(62.0),
        invert_normal=True,
        invert_scan=False,
    )


@pytest.fixture(scope="session")
def coinv_problem_inverted():
    return _build_coinv_problem(
        Lscan=10,
        Nscan=100,
        Wdevice=5,
        SNR=200,
        rho1_true=0.2,
        rho2_true=0.3,
        h_true=0.5,
        phi_true=np.deg2rad(62.0),
        invert_normal=True,
        invert_scan=True,
    )


@pytest.fixture
def make_coinverter():
    def _make(problem, **overrides):
        fwd, signal, _, params, _ = problem
        kwargs = {
            "initial_height": 1.0,
            "initial_phi_eff": np.deg2rad(75.0) * (-1 if params.invert_scan else 1),
            "regularizer": 1e-3 * backward.Tikhonov1(),
            "crossvalidation_size": 2,
            "lr_sensor": 3e-2,
            "device": torch.device("cpu"),
        }
        kwargs.update(overrides)  # any Coinversion1D argument; typos raise TypeError
        return backward.Coinversion1D(fwd, signal, **kwargs)

    return _make


@pytest.fixture(scope="session")
def oversampledinverter1d_with_random_currents() -> tuple[
    backward.Inverter1D, np.ndarray
]:
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    params = forward.Forward1DParameters(
        Lscan=10.0, Wdevice=5, Nscan=200, phi=np.deg2rad(phi)
    )

    y = params.ydevice.detach().cpu().numpy()
    Ngaussians = 5
    sigmas = rng.uniform(0.1, 0.5, size=Ngaussians)
    ymeans = rng.uniform(-2.5, 2.5, size=Ngaussians)
    amplitudes = rng.uniform(0.5, 1.5, size=Ngaussians)
    current = np.zeros_like(y)
    for sigma, ymean, amplitude in zip(sigmas, ymeans, amplitudes, strict=True):
        current += amplitude * np.exp(-((y - ymean) ** 2) / (2 * sigma**2))

    fwd = forward.Forward1D(params)
    signal_original = fwd.forward(current)
    yscan_original = params.yscan.detach().cpu().numpy()

    # Undersample the output signal on a coarser grid
    yscan = np.linspace(yscan_original.min(), yscan_original.max(), 195)
    signal = np.interp(yscan, yscan_original, signal_original)
    zeroregularizer = 0.0 * backward.Tikhonov1()

    inverter = backward.Inverter1DOversampled(fwd, yscan, signal, zeroregularizer)

    return inverter, current


@pytest.fixture(scope="session")
def inverter2d_with_uniform_current() -> tuple[backward.Inverter2D, np.ndarray]:
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    BM = SimplyConnectedCurrentMesh.BoundaryMarker
    vertices = [[-4, -1], [-4, 1], [4, 1], [4, -1]]
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segment_markers = [
        BM.NEUMANN_IN,
        BM.DIRICHLET_RIGHT,
        BM.NEUMANN_OUT,
        BM.DIRICHLET_LEFT,
    ]
    barmesh = SimplyConnectedCurrentMesh(
        vertices, segments, segment_markers, max_area=0.1
    )

    ymesh = barmesh.vertices[:, 1]
    streamfunction = ymesh - np.min(ymesh)  # Uniform current density across the bar

    params = forward.Forward2DParameters(Lx=8.0, Ly=6.0, Nx=200, Ny=150, phi=phi)
    meshtogrid = forward.MeshToGrid(barmesh, params)
    fwd = forward.Forward2DCurrent(params)

    signal = fwd.forward(*meshtogrid.streamfunction_to_currents(streamfunction))
    inverter = backward.Inverter2D(meshtogrid, fwd, signal)

    return inverter, streamfunction


def corner_block_geometry(
    W: float = 3.0,
    H: float = 3.0,
    lead_width: float = 0.6,
    lead_length: float = 2.0,
    corner_radius: float = 0.15,
    n_arc: int = 8,
) -> tuple[list[list[float]], list[list[int]], list[BM]]:
    """Square block with an inlet lead on one side and an outlet lead on the other."""
    w, L = lead_width, lead_length
    if not (0 < w < min(W, H)):
        raise ValueError("lead_width must be positive and smaller than the block.")

    # Counter-clockwise polygon: (point, marker of the segment starting here)
    corners = [
        ((-L, 0.0), BM.DIRICHLET_LEFT, False),  # inlet lead, bottom wall starts
        ((W, 0.0), BM.DIRICHLET_LEFT, True),  # block bottom-right (convex)
        ((W, H - w), BM.DIRICHLET_LEFT, True),  # re-entrant corner into outlet lead
        ((W + L, H - w), BM.NEUMANN_OUT, False),  # outlet (open end)
        ((W + L, H), BM.DIRICHLET_RIGHT, False),
        ((0.0, H), BM.DIRICHLET_RIGHT, True),  # block top-left (convex)
        ((0.0, w), BM.DIRICHLET_RIGHT, True),  # re-entrant corner into inlet lead
        ((-L, w), BM.NEUMANN_IN, False),  # inlet (open end)
    ]

    points, markers = [], []
    n = len(corners)
    for i, (p, marker_out, roundable) in enumerate(corners):
        p = np.asarray(p, dtype=float)
        if not roundable or corner_radius <= 0:
            points.append(p)
            markers.append(marker_out)
            continue

        # Fillet tangent to both edges: unit vectors towards the neighbouring corners
        p_prev = np.asarray(corners[i - 1][0], dtype=float)
        p_next = np.asarray(corners[(i + 1) % n][0], dtype=float)
        a = (p_prev - p) / np.linalg.norm(p_prev - p)
        b = (p_next - p) / np.linalg.norm(p_next - p)
        theta = np.arccos(np.clip(a @ b, -1.0, 1.0))  # angle between the edges
        t = corner_radius / np.tan(theta / 2)  # corner to tangent point
        if t > 0.5 * min(np.linalg.norm(p_prev - p), np.linalg.norm(p_next - p)):
            raise ValueError(f"corner_radius too large for the corner at {tuple(p)}.")
        centre = p + (a + b) / np.linalg.norm(a + b) * corner_radius / np.sin(theta / 2)
        T1, T2 = p + a * t, p + b * t
        phi1 = np.arctan2(T1[1] - centre[1], T1[0] - centre[0])
        phi2 = np.arctan2(T2[1] - centre[1], T2[0] - centre[0])
        dphi = (phi2 - phi1 + np.pi) % (2 * np.pi) - np.pi  # short way round
        for k in range(n_arc + 1):
            phi = phi1 + dphi * k / n_arc
            points.append(centre + corner_radius * np.array([np.cos(phi), np.sin(phi)]))
            markers.append(
                marker_out
            )  # both edges at a rounded corner are the same wall

    # Mesh coordinates [x, y] = [h, u]
    vertices = [[h, u] for h, u in points]
    segments = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]

    cx, cy = np.mean(vertices, axis=0)
    vertices = [[x - cx, y - cy] for x, y in vertices]
    return vertices, segments, markers


def rho_ring(
    m: SimplyConnectedCurrentMesh, r: float = 0.6, sigma: float = 0.5, a: float = 1.5
) -> np.ndarray:
    """Resistivity per triangle: elliptical ring in a resistive block."""
    xt, yt = m.centers.T
    R_eff = np.sqrt(xt**2 + (yt / a) ** 2)
    return 10 - 9 * np.exp(-((R_eff - r) ** 2) / (2 * sigma**2))


def _coinv2d_params(invert_scan: bool) -> forward.Forward2DParameters:
    # Corner block spans ~8 x 3 around the origin; window adds >= 2 um margin per side.
    # Square pixels: 12/256 = 6/128.
    return forward.Forward2DParameters(
        Lx=6.0,
        Ly=6.0,
        Nx=128,
        Ny=128,
        rho1=0.1,
        rho2=0.24,
        height=0.35,
        phi=np.deg2rad(63.0),
        invert_normal=True,
        invert_scan=invert_scan,
    )


@dataclasses.dataclass(frozen=True)
class Coinv2DProblem:
    currentforward: forward.Forward2DCurrent
    meshtogrid: forward.MeshToGrid  # fit mesh -> goes into Coinversion2D
    data: np.ndarray
    params: forward.Forward2DParameters
    noise_sigma: float
    I_total_true: float
    streamfunction_true: (
        np.ndarray | None
    )  # on fit-mesh vertices, only if fit_mesh is truth_mesh
    truth_meshtogrid: forward.MeshToGrid  # diagnostics only
    truth_streamfunction: np.ndarray  # diagnostics only


def _build_coinv2d_problem(
    truth_mesh: SimplyConnectedCurrentMesh,
    fit_mesh: SimplyConnectedCurrentMesh,
    params: forward.Forward2DParameters,
    rho: np.ndarray | None = None,
    I_total: float = 1.0,
    SNR: float = 200.0,
    seed: int = 0,
) -> Coinv2DProblem:
    assert np.isclose(params.Lx / params.Nx, params.Ly / params.Ny), (
        "pixels must be square"
    )
    if rho is None:
        rho = np.ones(len(truth_mesh.centers))
    assert rho.shape == (len(truth_mesh.centers),)

    # True current on the truth mesh
    res = InhomogeneousOhmicSolver(truth_mesh, rho).solve(left_bc=0.0, right_bc=I_total)
    truth_streamfunction = res["streamfunction"]

    # Clean signal: streamfunction -> gridded currents -> forward model\
    # All on the params grid
    truth_meshtogrid = forward.MeshToGrid(truth_mesh, params)
    currentforward = forward.Forward2DCurrent(params)
    signal = currentforward.forward(
        *truth_meshtogrid.streamfunction_to_currents(truth_streamfunction)
    )

    # Noisy data
    rng = np.random.default_rng(seed)
    noise_sigma = float(np.ptp(signal) / SNR)
    data = signal + rng.normal(0.0, noise_sigma, size=signal.shape)

    return Coinv2DProblem(
        currentforward=currentforward,
        meshtogrid=forward.MeshToGrid(fit_mesh, params),
        data=data,
        params=params,
        noise_sigma=noise_sigma,
        I_total_true=I_total,
        streamfunction_true=truth_streamfunction if fit_mesh is truth_mesh else None,
        truth_meshtogrid=truth_meshtogrid,
        truth_streamfunction=truth_streamfunction,
    )


@pytest.fixture(scope="session")
def corner_mesh_coarse() -> SimplyConnectedCurrentMesh:
    vertices, segments, markers = corner_block_geometry(W=2.0, H=3.0, lead_length=3.0)
    return SimplyConnectedCurrentMesh(
        vertices, segments, markers, max_area=4e-2, refine_distance=0.2
    )


@pytest.fixture(scope="session")
def corner_mesh_fine() -> SimplyConnectedCurrentMesh:
    vertices, segments, markers = corner_block_geometry(W=2.0, H=3.0, lead_length=3.0)
    return SimplyConnectedCurrentMesh(
        vertices, segments, markers, max_area=4e-4, refine_distance=0.01
    )


@pytest.fixture(scope="session")
def corner_mesh_coarse_perturbed() -> SimplyConnectedCurrentMesh:
    vertices, segments, markers = corner_block_geometry(W=2.0, H=3.0, lead_length=3.0)
    rng = np.random.default_rng(42)
    sigma = 0.01
    offsetx, offsety = rng.uniform(-0.05, 0.05, size=2)
    vertices_perturbed = [
        [x + rng.normal(0.0, sigma) + offsetx, y + rng.normal(0.0, sigma) + offsety]
        for x, y in vertices
    ]
    return SimplyConnectedCurrentMesh(
        vertices_perturbed, segments, markers, max_area=4e-2, refine_distance=0.2
    )


@pytest.fixture(scope="session")
def coinv2d_problem(corner_mesh_coarse) -> Coinv2DProblem:
    """Trivial problem: homogeneous, truth and fit on the same coarse mesh."""
    return _build_coinv2d_problem(
        truth_mesh=corner_mesh_coarse,
        fit_mesh=corner_mesh_coarse,
        params=_coinv2d_params(invert_scan=False),
    )


@pytest.fixture(scope="session")
def coinv2d_problem_inverted(corner_mesh_coarse) -> Coinv2DProblem:
    """Trivial problem with invert_scan=True."""
    return _build_coinv2d_problem(
        truth_mesh=corner_mesh_coarse,
        fit_mesh=corner_mesh_coarse,
        params=_coinv2d_params(invert_scan=True),
    )


@pytest.fixture(scope="session")
def coinv2d_problem_mismatch(
    corner_mesh_fine, corner_mesh_coarse_perturbed
) -> Coinv2DProblem:
    """Realistic problem: inhomogeneous truth on a fine mesh, fit on a coarse mesh."""
    return _build_coinv2d_problem(
        truth_mesh=corner_mesh_fine,
        fit_mesh=corner_mesh_coarse_perturbed,
        params=_coinv2d_params(invert_scan=False),
        rho=rho_ring(corner_mesh_fine),
    )


@pytest.fixture
def make_coinverter2d():
    def _make(problem: Coinv2DProblem, **overrides) -> backward.Coinversion2D:
        kwargs = {
            "lam1": 1e-11,
            "lam2": 1e-10,
            "lam3": 1e-15,
            "lumped_mass": True,
            "initial_height": 1.0,
            "initial_phi_eff": np.deg2rad(45.0)
            * (-1 if problem.params.invert_scan else 1),
            "crossvalidation_size": 2,
            "lr_current": 1.0,
            "cache_kernels": True,
            "use_cg": True,
            "device": torch.device("cpu"),
        }
        kwargs.update(overrides)  # any Coinversion2D argument; typos raise TypeError
        return backward.Coinversion2D(
            problem.meshtogrid, problem.currentforward, problem.data, **kwargs
        )

    return _make


@pytest.fixture
def single_thread():
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n_threads)

@pytest.fixture(scope="session")
def disk_mesh_freestanding() -> SimplyConnectedFreestandingMesh:
    """Unit disk, curved boundary; q30 guarantees obtuse triangles."""
    n_boundary = 64
    phi = 2 * np.pi * np.arange(n_boundary) / n_boundary
    vertices = [[np.cos(p), np.sin(p)] for p in phi]
    segments = [[i, (i + 1) % n_boundary] for i in range(n_boundary)]
    return SimplyConnectedFreestandingMesh(vertices, segments, max_area=2e-3)