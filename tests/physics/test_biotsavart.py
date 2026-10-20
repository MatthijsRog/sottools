import numpy as np
import pytest
from scipy.signal import fftconvolve

import sottools.forward as forward
import sottools.mesh as mesh


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_1d_forward_point(seed):
    # For a very small SQUID, see if the 1D code accurate produces a 1/R Biot-Savart
    # pattern

    R = 0.0005
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)
    H = rng.uniform(0.5, 5.0)
    Lscan = rng.uniform(5.0, 15.0)
    N = 500
    W = Lscan / N * 2  # 2 point current strip
    area = np.pi * R**2

    params = forward.Forward1DParameters(
        Lscan=Lscan,
        Wdevice=W,
        Nscan=500,
        rho1=R,
        rho2=(1 + 1e-6) * R,
        height=H,
        phi=phi,
    )
    currents = np.ones(2)
    fwd = forward.Forward1D(params)
    signal = fwd.forward(currents) / area

    y = params.yscan.numpy()
    signal_expected = (
        2e-1
        * W
        * 1
        / np.sqrt(params.height**2 + y**2)
        * (
            y / np.sqrt(y**2 + params.height**2) * np.cos(phi)
            - params.height / np.sqrt(y**2 + params.height**2) * np.sin(phi)
        )
    )

    assert np.allclose(signal, signal_expected, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_1d_extended_squid(seed):
    # Test if the 1D code + manual convoltution is identical to running the 1D code
    # for an extended SQUID
    rng = np.random.default_rng(seed)
    Lscan = rng.uniform(5.0, 15.0)
    Nscan = 500
    W = Lscan / Nscan * 2  # 2 point current strip
    R = rng.uniform(0.3, 1.0)
    H = rng.uniform(0.5, 5.0)

    params_big = forward.Forward1DParameters(
        Lscan=Lscan, Wdevice=W, Nscan=Nscan, rho1=R, rho2=(1 + 1e-6) * R, height=H
    )
    currents = np.ones(2)
    fwd_big = forward.Forward1D(params_big)
    signalbig = fwd_big.forward(currents)

    params_point = forward.Forward1DParameters(
        Lscan=Lscan,
        Wdevice=W,
        Nscan=Nscan,
        rho1=0.0005,
        rho2=(1 + 1e-6) * 0.0005,
        height=H,
    )
    currents = np.ones(2)
    fwd_point = forward.Forward1D(params_point)
    signalpoint = fwd_point.forward(currents)

    areadiff = (params_big.rho1**2) / (params_point.rho1**2)

    y = params_big.yscan
    R = float(params_big.rho1)

    def F(y):
        yc = np.clip(y, -R, R)
        return (yc * np.sqrt(R**2 - yc**2) + R**2 * np.arcsin(yc / R)) / (np.pi * R**2)

    dy = float(y[1] - y[0])
    Nk = 2 * int(np.ceil(R / dy)) + 1
    edges = (np.arange(Nk + 1) - Nk / 2) * dy
    kernel = F(edges[1:]) - F(edges[:-1])

    signalbig_smooth = np.convolve(signalpoint * areadiff, kernel, mode="same")
    N = int(round(params_big.rho1 / (params_big.Lscan / params_big.Nscan)))

    assert np.allclose(signalbig[N:-N], signalbig_smooth[N:-N], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_2d_linecurrent(seed):
    rng = np.random.default_rng(seed)

    dx = 0.05
    Nx = 12 * rng.integers(
        50,
        100,
    )
    Ny = Nx // 6
    Lx = dx * Nx
    Ly = dx * Ny
    R = 0.0005
    H = rng.uniform(0.5, 5.0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    params = forward.Forward2DParameters(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        rho1=R,
        rho2=(1 + 1e-6) * R,
        height=H,
        phi=phi,
    )

    Jx = np.zeros_like(params.xx.numpy())
    Jy = np.zeros_like(params.xx.numpy())
    Jx[:, Ny // 2] = 1.0

    fwd = forward.Forward2DCurrent(params)
    signal = fwd.forward(Jx, Jy)

    y = params.yy.numpy()[0, :]
    width = float(params.Lx / params.Nx)  # Exactly 1 cell width in y-direction
    Curr = 1.0 * width
    s_expected = (
        2
        * 1e-1
        * Curr
        * 1
        / np.sqrt(H**2 + y**2)
        * (
            y / np.sqrt(y**2 + H**2) * np.cos(phi)
            - H / np.sqrt(y**2 + H**2) * np.sin(phi)
        )
    )

    area = np.pi * R**2

    assert np.allclose(signal[100, :] / area, s_expected, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_2d_extended_squid(seed):
    # Test if the 2D code + manual convoltution is identical to running the 2D code
    # for an extended SQUID
    rng = np.random.default_rng(seed)
    R = rng.uniform(0.3, 1.0)
    dx = 0.05
    Nx = 12 * rng.integers(
        50,
        100,
    )
    Ny = Nx // 6
    Lx = dx * Nx
    Ly = dx * Ny
    H = rng.uniform(0.5, 5.0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    params_point = forward.Forward2DParameters(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        rho1=0.0005,
        rho2=(1 + 1e-6) * 0.0005,
        height=H,
        phi=phi,
    )
    params_big = forward.Forward2DParameters(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, rho1=R, rho2=(1 + 1e-6) * R, height=H, phi=phi
    )

    Jx = np.zeros_like(params_point.xx.numpy())
    Jy = np.zeros_like(Jx)
    Jx[:, Ny // 2] = 1.0

    fwd_point = forward.Forward2DCurrent(params_point)
    signal_point = fwd_point.forward(Jx, Jy)

    fwd_big = forward.Forward2DCurrent(params_big)
    signal_big = fwd_big.forward(Jx, Jy)

    areadiff = (R**2) / (params_point.rho1**2)
    signal_point_scaled = np.asarray(signal_point) * areadiff
    signal_big = np.asarray(signal_big)

    # Disk kernel via subpixel sampling
    dx = float(params_point.Lx / params_point.Nx)
    dy = float(params_point.Ly / params_point.Ny)
    Nkx = 2 * int(np.ceil(R / dx)) + 1
    Nky = 2 * int(np.ceil(R / dy)) + 1
    nsub = 16
    kernel = np.zeros((Nkx, Nky))
    for i in range(Nkx):
        for j in range(Nky):
            cx = (i - Nkx // 2) * dx
            cy = (j - Nky // 2) * dy
            sx = cx + (np.arange(nsub) - (nsub - 1) / 2) / nsub * dx
            sy = cy + (np.arange(nsub) - (nsub - 1) / 2) / nsub * dy
            SX, SY = np.meshgrid(sx, sy)
            kernel[i, j] = np.mean(SX**2 + SY**2 <= R**2)
    kernel /= kernel.sum()

    signal_conv = fftconvolve(signal_point_scaled, kernel, mode="same")

    N = int(np.ceil(R / min(dx, dy))) + 1
    interior = (slice(N, -N), slice(N, -N))
    assert np.allclose(
        signal_big[interior], signal_conv[interior], rtol=1e-2, atol=1e-3
    )


def test_height_dependence():
    # Test that the height dependence of the 1D and 2D forward models follows 1/H
    # in the case all measured field is in plane

    # First 1D:

    heights = np.logspace(-0.5, 2, 11)
    peaks = []
    R = 0.0005
    for h in heights:
        params1d = forward.Forward1DParameters(
            Lscan=10.0,
            Wdevice=0.04,
            Nscan=500,
            rho1=R,
            rho2=(1 + 1e-6) * R,
            height=h,
            phi=0.99999999 * np.pi / 2,
            invert_normal=True,
        )
        currents = np.ones(2)
        fwd1d = forward.Forward1D(params1d)
        signal1d = fwd1d.forward(currents)
        peaks.append(np.max(signal1d) / (np.pi * R**2))

    peaks = np.array(peaks)
    expected_peaks = 2 * 1e-1 * 0.04 / heights

    assert np.allclose(peaks, expected_peaks, rtol=1e-2, atol=1e-9)

    # Now 2D
    peaks = []
    heights = np.logspace(-0.5, 1.1, 11)  # Smaller range because of the limited FOV
    for h in heights:
        params2d = forward.Forward2DParameters(
            Lx=50.0,
            Ly=5.0,
            Nx=500,
            Ny=50,
            rho1=R,
            rho2=(1 + 1e-6) * R,
            height=h,
            phi=0.99999999 * np.pi / 2,
            invert_normal=True,
        )
        Jx = np.zeros((params2d.Nx, params2d.Ny))
        Jy = np.zeros((params2d.Nx, params2d.Ny))
        Jx[:, params2d.Ny // 2] = 1.0
        fwd2d = forward.Forward2DCurrent(params2d)
        signal2d = fwd2d.forward(Jx, Jy)
        peaks.append(np.max(signal2d) / (np.pi * R**2))

    peaks = np.array(peaks)
    expected_peaks = 2 * 1e-1 * 0.1 / heights
    assert np.allclose(peaks, expected_peaks, rtol=1e-1, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mesh_to_grid_strip(seed):
    rng = np.random.default_rng(0)

    dx = 0.05
    Nx = 12 * rng.integers(50, 100)
    Ny = Nx // 6
    Lx = dx * Nx
    Ly = dx * Ny
    R = 0.0005
    H = rng.uniform(0.5, 5.0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    W = Ly / Ny  # one pixel wide — resolves correctly

    vertices = [[-Lx / 2, -W / 2], [-Lx / 2, W / 2], [Lx / 2, W / 2], [Lx / 2, -W / 2]]
    segments = [[0, 1], [1, 2], [2, 3], [3, 0]]
    segment_markers = [
        mesh.SimplyConnectedCurrentMesh.BoundaryMarker.NEUMANN_IN,
        mesh.SimplyConnectedCurrentMesh.BoundaryMarker.DIRICHLET_RIGHT,
        mesh.SimplyConnectedCurrentMesh.BoundaryMarker.NEUMANN_OUT,
        mesh.SimplyConnectedCurrentMesh.BoundaryMarker.DIRICHLET_LEFT,
    ]
    strip = mesh.SimplyConnectedCurrentMesh(
        vertices, segments, segment_markers, max_area=0.0001, refine_distance=0.1
    )
    streamfunc = strip.vertices[:, 1]

    fwdparams = forward.Forward2DParameters(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        rho1=R,
        rho2=(1 + 1e-6) * R,
        height=H,
        phi=phi,
    )

    mesh2grid = forward.MeshToGrid(strip, fwdparams)
    fwd = forward.Forward2DCurrent(fwdparams)
    signal = fwd.forward(*mesh2grid.streamfunction_to_currents(streamfunc))

    area = np.pi * R**2
    Curr = np.max(streamfunc) - np.min(streamfunc)
    y = fwdparams.yy.numpy()[0, :]
    s_expected = (
        2
        * 1e-1
        * Curr
        * 1
        / np.sqrt(H**2 + y**2)
        * (
            y / np.sqrt(y**2 + H**2) * np.cos(phi)
            - H / np.sqrt(y**2 + H**2) * np.sin(phi)
        )
    )

    assert np.allclose(signal[Nx // 2, :] / area, s_expected, rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("phi", [0.0, 0.3, -0.8, 1.2, 2.8])
def test_2d_biotsavart_bruteforce(phi):
    """Finite, tilted, tapered loop vs direct Biot-Savart of a few current elements."""
    N, dx = 80, 0.25
    h, rho1, rho2 = 1.0, 0.4, 0.8
    params = forward.Forward2DParameters(
        Lx=N * dx, Ly=N * dx, Nx=N, Ny=N, rho1=rho1, rho2=rho2, height=h, phi=phi
    )

    # A few random current elements near the centre, with both components
    rng = np.random.default_rng(0)
    Jx, Jy = np.zeros((N, N)), np.zeros((N, N))
    ix, iy = rng.integers(N // 2 - 4, N // 2 + 4, size=(2, 6))
    Jx[ix, iy] = rng.uniform(-1, 1, 6)
    Jy[ix, iy] = rng.uniform(-1, 1, 6)

    signal = forward.Forward2DCurrent(params).forward(Jx, Jy)

    # Sources as point current elements at pixel centres
    X, Y = params.xx.numpy().astype(float), params.yy.numpy().astype(float)
    isrc = np.nonzero(Jx**2 + Jy**2)
    r_src = np.column_stack([X[isrc], Y[isrc], np.zeros(len(isrc[0]))])
    J_src = np.column_stack([Jx[isrc], Jy[isrc], np.zeros(len(isrc[0]))])

    # Loop: disk tilted about x by phi, in-plane axes x_hat and e, normal n
    e = np.array([0.0, np.cos(phi), -np.sin(phi)])
    n = np.array([0.0, np.sin(phi), np.cos(phi)])

    # Quadrature weights: trapezoid in alpha; Gauss-Legendre in rho split at rho1.
    # Each weight includes rho * W(rho).
    n_alpha = 256
    alpha = 2 * np.pi * np.arange(n_alpha) / n_alpha
    x, w = np.polynomial.legendre.leggauss(32)
    r_in = 0.5 * rho1 * (1 + x)
    r_out = rho1 + 0.5 * (rho2 - rho1) * (1 + x)
    rho = np.concatenate([r_in, r_out])
    w_rho = np.concatenate(
        [
            0.5 * rho1 * w * r_in,
            0.5 * (rho2 - rho1) * w * r_out * (rho2 - r_out) / (rho2 - rho1),
        ]
    )
    A, R = np.meshgrid(alpha, rho, indexing="ij")
    dA = (2 * np.pi / n_alpha) * w_rho[None, :]
    loop = R[..., None] * (
        np.cos(A)[..., None] * np.array([1.0, 0, 0]) + np.sin(A)[..., None] * e
    )  # (Na, Nr, 3)

    # Direct Biot-Savart flux at SQUID positions in the central patch
    patch = slice(N // 2 - 12, N // 2 + 12, 2)
    ref = np.zeros((N, N))
    for i in range(N)[patch]:
        for j in range(N)[patch]:
            r0 = np.array([X[i, j], Y[i, j], h])
            p = r0 + loop[:, :, None, :] - r_src  # (Na, Nr, Ns, 3)
            integrand = (np.cross(J_src, p) @ n) / np.linalg.norm(p, axis=-1) ** 3
            ref[i, j] = (
                1e-1 * dx**2 * np.sum(dA[..., None] * integrand)
            )  # mu0/4pi = 0.1

    err = np.max(np.abs(signal[patch, patch] - ref[patch, patch]))
    assert err < 1e-2 * np.max(np.abs(ref[patch, patch]))
