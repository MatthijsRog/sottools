import numpy as np
import pytest
import torch

import sottools.backward as backward
import sottools.forward as forward
from sottools.mesh import SimplyConnectedCurrentMesh

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
