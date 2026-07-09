import numpy as np
import pytest

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

    inverter = backward.Inverter1D(fwd, signal)

    return inverter, current


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

    inverter = backward.Inverter1DOversampled(fwd, yscan, signal)

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
