import re

import numpy as np
import pytest
import torch

import sottools.forward as forward


def test_forwardparametervalidation():
    with pytest.raises(ValueError, match="Lscan must be positive and nonzero."):
        forward.Forward1DParameters(Lscan=0.0, Wdevice=1.0, Nscan=32)

    with pytest.raises(
        ValueError, match=re.escape("Gamma must be strictly inside (-pi/2, pi/2).")
    ):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=32, gamma=np.pi / 2)

    with pytest.raises(
        ValueError, match=re.escape("Gamma must be strictly inside (-pi/2, pi/2).")
    ):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=32, gamma=-np.pi / 2)

    with pytest.raises(
        ValueError, match=re.escape("Gamma must be strictly inside (-pi/2, pi/2).")
    ):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=32, gamma=-20.0)

    with pytest.raises(
        ValueError, match=re.escape("Wdevice must be less than Lscan/cos(gamma).")
    ):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=2.0, Nscan=32, gamma=np.pi / 4)

    with pytest.raises(ValueError, match="Wdevice must be positive."):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=-1.0, Nscan=32, gamma=0.0)

    with pytest.raises(ValueError, match="Nscan must be positive and non-zero."):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=0, gamma=0.0)

    with pytest.raises(ValueError, match="Nscan must be positive and non-zero."):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=-10, gamma=0.0)

    with pytest.raises(ValueError, match="Nscan must be an even integer."):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=10.5, gamma=0.0)

    with pytest.raises(ValueError, match="Nscan must be an even integer."):
        forward.Forward1DParameters(Lscan=1.0, Wdevice=1.0, Nscan=11, gamma=0.0)

    with pytest.raises(ValueError, match="Nx must be positive and non-zero."):
        forward.Forward2DParameters(Nx=0)

    with pytest.raises(ValueError, match="Ny must be positive and non-zero."):
        forward.Forward2DParameters(Ny=0)

    with pytest.raises(ValueError, match="Nx must be positive and non-zero."):
        forward.Forward2DParameters(Nx=-5)

    with pytest.raises(ValueError, match="Ny must be positive and non-zero."):
        forward.Forward2DParameters(Ny=-5)

    with pytest.raises(ValueError, match="Nx must be even."):
        forward.Forward2DParameters(Nx=11)

    with pytest.raises(ValueError, match="Ny must be even."):
        forward.Forward2DParameters(Ny=11)

    with pytest.raises(
        ValueError,
        match="Pixels must be square, so Ny must be set such that Ly/Ny = Lx/Nx.",
    ):
        forward.Forward2DParameters(Nx=10, Ny=12, Lx=1.0, Ly=1.0)


def test_forwardparameterdefaults():
    # All properties must be accessible and sensible upon default values
    params1d = forward.Forward1DParameters()

    assert isinstance(params1d.Lscan, float)
    assert params1d.Lscan > 0
    assert not np.isnan(params1d.Lscan)

    assert isinstance(params1d.gamma, float)
    assert -np.pi / 2 < params1d.gamma < np.pi / 2

    assert isinstance(params1d.rho1, float)
    assert params1d.rho1 > 0
    assert params1d.rho1 <= params1d.rho2
    assert not np.isnan(params1d.rho1)

    assert isinstance(params1d.rho2, float)
    assert params1d.rho2 > 0
    assert params1d.rho2 >= params1d.rho1
    assert not np.isnan(params1d.rho2)

    assert isinstance(params1d.phi, float)
    assert 0 <= params1d.phi <= np.pi

    assert isinstance(params1d.height, float)
    assert params1d.height > np.sin(params1d.phi) * params1d.rho2
    assert not np.isnan(params1d.height)

    assert isinstance(params1d.dyscan, float)
    assert np.isclose(params1d.dyscan, params1d.Lscan / params1d.Nscan)

    assert isinstance(params1d.dyproj, float)
    assert np.isclose(params1d.dyproj, params1d.dyscan * np.cos(params1d.gamma))

    assert isinstance(params1d.Ndevice, int)
    assert params1d.Ndevice == int(round(params1d.Wdevice / params1d.dyproj))

    assert isinstance(params1d.Lproj, float)
    assert np.isclose(params1d.Lproj, params1d.Nscan * params1d.dyproj)

    assert isinstance(params1d.Lscan, float)
    assert params1d.Lscan > 0
    assert not np.isnan(params1d.Lscan)

    assert isinstance(params1d.Nscan, int)
    assert params1d.Nscan > 0
    assert params1d.Nscan % 2 == 0

    assert isinstance(params1d.Wdevice, float)
    assert params1d.Wdevice > 0
    assert params1d.Wdevice < params1d.Lscan / np.cos(params1d.gamma)
    assert not np.isnan(params1d.Wdevice)

    assert isinstance(params1d.yscan, torch.Tensor)
    assert len(params1d.yscan) == params1d.Nscan
    assert np.isclose(float(params1d.yscan[0]), -params1d.Lscan / 2)
    assert np.isclose(float(params1d.yscan[-1]), params1d.Lscan / 2 - params1d.dyscan)

    assert isinstance(params1d.yproj, torch.Tensor)
    assert len(params1d.yproj) == params1d.Nscan
    assert np.isclose(float(params1d.yproj[0]), -params1d.Lproj / 2)
    assert np.isclose(float(params1d.yproj[-1]), params1d.Lproj / 2 - params1d.dyproj)

    assert isinstance(params1d.ydevice, torch.Tensor)
    assert len(params1d.ydevice) == params1d.Ndevice
    assert np.isclose(float(params1d.ydevice[0]), -params1d.Wdevice / 2)
    assert np.isclose(
        float(params1d.ydevice[-1]), params1d.Wdevice / 2 - params1d.dyproj
    )

    assert isinstance(params1d.phi_eff, float)
    assert params1d.phi_eff == params1d.phi

    assert isinstance(params1d.kernel_sign, float)
    assert params1d.kernel_sign == 1.0

    params2d = forward.Forward2DParameters()

    assert isinstance(params2d.Lx, float)
    assert params2d.Lx > 0
    assert not np.isnan(params2d.Lx)

    assert isinstance(params2d.Ly, float)
    assert params2d.Ly > 0
    assert not np.isnan(params2d.Ly)

    assert isinstance(params2d.Nx, int)
    assert params2d.Nx > 0
    assert params2d.Nx % 2 == 0

    assert isinstance(params2d.Ny, int)
    assert params2d.Ny > 0
    assert params2d.Ny % 2 == 0

    assert isinstance(params2d.rho1, float)
    assert params2d.rho1 > 0
    assert params2d.rho1 <= params2d.rho2
    assert not np.isnan(params2d.rho1)

    assert isinstance(params2d.rho2, float)
    assert params2d.rho2 > 0
    assert params2d.rho2 >= params2d.rho1
    assert not np.isnan(params2d.rho2)

    assert isinstance(params2d.phi, float)
    assert 0 <= params2d.phi <= np.pi

    assert isinstance(params2d.height, float)
    assert params2d.height > np.sin(params2d.phi) * params2d.rho2
    assert not np.isnan(params2d.height)

    assert isinstance(params2d.x, torch.Tensor)
    assert len(params2d.x) == params2d.Nx
    assert np.isclose(float(params2d.x[0]), -params2d.Lx / 2)
    assert np.isclose(
        float(params2d.x[-1]), params2d.Lx / 2 - params2d.Lx / params2d.Nx
    )

    assert isinstance(params2d.y, torch.Tensor)
    assert len(params2d.y) == params2d.Ny
    assert np.isclose(float(params2d.y[0]), -params2d.Ly / 2)
    assert np.isclose(
        float(params2d.y[-1]), params2d.Ly / 2 - params2d.Ly / params2d.Ny
    )

    assert isinstance(params2d.xx, torch.Tensor)
    assert params2d.xx.shape == (params2d.Nx, params2d.Ny)
    assert np.isclose(float(params2d.xx[0, 0]), -params2d.Lx / 2)
    assert np.isclose(
        float(params2d.xx[-1, -1]), params2d.Lx / 2 - params2d.Lx / params2d.Nx
    )

    assert isinstance(params2d.yy, torch.Tensor)
    assert params2d.yy.shape == (params2d.Nx, params2d.Ny)
    assert np.isclose(float(params2d.yy[0, 0]), -params2d.Ly / 2)
    assert np.isclose(
        float(params2d.yy[-1, -1]), params2d.Ly / 2 - params2d.Ly / params2d.Ny
    )


def test_forwardaxes():
    params1d = forward.Forward1DParameters(Lscan=10.0, Wdevice=5.0, Nscan=100)

    # Test that all y-axes are of the right size, and also that they are
    # centered around zero
    assert len(params1d.yscan) == 100
    assert np.isclose(float(params1d.yscan[50]), 0.0, atol=1e-6)
    assert len(params1d.ydevice) == 50
    assert np.isclose(float(params1d.ydevice[25]), 0.0, atol=1e-6)
    assert len(params1d.yproj) == 100
    assert np.isclose(float(params1d.yproj[50]), 0.0, atol=1e-6)

    params2d = forward.Forward2DParameters(
        Lx=4.0,
        Ly=2.0,
        Nx=100,
        Ny=50,
    )

    # Test that all x- and y-axes are of the right size, and also that they are
    # centered around zero
    assert len(params2d.x) == 100
    assert np.isclose(float(params2d.x[50]), 0.0, atol=1e-6)
    assert len(params2d.y) == 50
    assert np.isclose(float(params2d.y[25]), 0.0, atol=1e-6)
    assert params2d.xx.shape == (100, 50)
    assert np.isclose(float(params2d.xx[50, 25]), 0.0, atol=1e-6)
    assert params2d.yy.shape == (100, 50)
    assert np.isclose(float(params2d.yy[50, 25]), 0.0, atol=1e-6)


def test_forwardvalidity():
    # Bare-bones forward models should transform zero -> zero and produce
    # the correct shapes.

    params1d = forward.Forward1DParameters(Lscan=10.0, Wdevice=5.0, Nscan=100)
    fwd1d = forward.Forward1D(params1d)
    current1d = np.zeros(50)
    signal1d = fwd1d.forward(current1d)
    assert signal1d.shape == (100,)
    assert np.allclose(signal1d, 0.0)

    params2d = forward.Forward2DParameters(Lx=4.0, Ly=2.0, Nx=100, Ny=50)
    fwd2d = forward.Forward2DCurrent(params2d)
    current2d = np.zeros((100, 50))
    signal2d = fwd2d.forward(current2d, current2d)
    assert signal2d.shape == (100, 50)
    assert np.allclose(signal2d, 0.0)


def test_meshtogridshapes(bar_mesh_london):
    x0 = np.min(bar_mesh_london.vertices[:, 0])
    x1 = np.max(bar_mesh_london.vertices[:, 0])
    y0 = np.min(bar_mesh_london.vertices[:, 1])
    y1 = np.max(bar_mesh_london.vertices[:, 1])
    Lx = x1 - x0
    Ly = y1 - y0
    dx = 0.1
    Nx = int(np.ceil(Lx / dx))
    Ny = int(np.ceil(Ly / dx))

    params2d = forward.Forward2DParameters(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    meshtogrid = forward.MeshToGrid(bar_mesh_london, params2d)
    g_zero = np.zeros(len(bar_mesh_london.vertices))
    Kx, Ky = meshtogrid.streamfunction_to_currents(g_zero)

    assert Kx.shape == (Nx, Ny)
    assert Ky.shape == (Nx, Ny)
    assert np.allclose(Kx, 0.0, atol=np.max(g_zero) * 1e-5)
    assert np.allclose(Ky, 0.0, atol=np.max(g_zero) * 1e-5)

    # Now also test for a non-zero streamfunction that is constant
    g_const = np.ones(len(bar_mesh_london.vertices))
    Kx_const, Ky_const = meshtogrid.streamfunction_to_currents(g_const)
    assert Kx_const.shape == (Nx, Ny)
    assert Ky_const.shape == (Nx, Ny)
    assert np.allclose(Kx_const, 0.0, atol=np.max(g_const) * 1e-5)
    assert np.allclose(Ky_const, 0.0, atol=np.max(g_const) * 1e-5)


def test_forward2dwindowshape():
    params2d = forward.Forward2DParameters(Lx=4.0, Ly=2.0, Nx=100, Ny=50)
    fwd2d = forward.Forward2DCurrent(params2d)

    # The window should have the shape 3*Nx, 3*Ny
    window = fwd2d._window
    assert window.shape == (3 * params2d.Nx, 3 * params2d.Ny)

    # It should be unity in the center (Nx, Ny) and taper to zero at the edges
    assert np.allclose(
        window[params2d.Nx : 2 * params2d.Nx, params2d.Ny : 2 * params2d.Ny],
        1.0,
        atol=1e-6,
    )
    assert np.allclose(window[0, :], 0.0, atol=1e-6)  # Top edge
    assert np.allclose(window[-1, :], 0.0, atol=1e-6)  # Bottom edge
    assert np.allclose(window[:, 0], 0.0, atol=1e-6)  # Left edge
    assert np.allclose(window[:, -1], 0.0, atol=1e-6)  # Right edge


def test_currentextension():
    # A test function for the current must be properly extended to (Nx,Ny)
    # using proper mirroring and windowing. At the edge of the FOV, there must
    # be perfect alignment. And the window must taper to zero correctly.
    rng = np.random.default_rng(0)
    (a, b, c) = rng.uniform(-5, 5, size=3)

    params2d = forward.Forward2DParameters(Lx=4.0, Ly=2.0, Nx=100, Ny=50)
    fwd2d = forward.Forward2DCurrent(params2d)

    xx, yy = params2d.xx, params2d.yy
    test_func = a + b * xx + c * yy
    test_func_extended = fwd2d._extend_current(test_func).detach().cpu().numpy()

    assert test_func_extended.shape == (3 * params2d.Nx, 3 * params2d.Ny)
    assert np.allclose(
        test_func_extended[
            params2d.Nx : 2 * params2d.Nx, params2d.Ny : 2 * params2d.Ny
        ],
        test_func,
        atol=1e-6,
    )
    assert np.allclose(
        test_func_extended[params2d.Nx - 1, params2d.Ny : 2 * params2d.Ny],
        test_func[0, :],
        atol=1e-6,
    )
    assert np.allclose(
        test_func_extended[params2d.Nx : 2 * params2d.Nx, params2d.Ny - 1],
        test_func[:, 0],
        atol=1e-6,
    )
    assert np.allclose(
        test_func_extended[params2d.Nx * 2, params2d.Ny : 2 * params2d.Ny],
        test_func[-1, :],
        atol=1e-6,
    )
    assert np.allclose(
        test_func_extended[params2d.Nx : 2 * params2d.Nx, params2d.Ny * 2],
        test_func[:, -1],
        atol=1e-6,
    )
    assert np.allclose(test_func_extended[0, :], 0.0, atol=1e-6)
    assert np.allclose(test_func_extended[-1, :], 0.0, atol=1e-6)
    assert np.allclose(test_func_extended[:, 0], 0.0, atol=1e-6)
    assert np.allclose(test_func_extended[:, -1], 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "phi,gamma", [(0.0, 0.0), (0.4, 0.0), (0.3, 0.5), (-0.6, -0.3)]
)
def test_invertflags_fullstack_1d(phi: float, gamma: float) -> None:
    """Full-stack contract of the invert flags at arbitrary phi and gamma.

    invert_normal:  S -> -S.
    invert_scan:    S_scan(j) == reverse(S_base(reverse(j)))  (mirrored world).

    The reversal identity is exact on the discrete grid provided that
    n_alpha is even (alpha -> alpha + pi maps the quadrature onto itself)
    and Nscan - Ndevice is even (zero-padding symmetric under reversal).
    """

    def make(invert_normal: bool, invert_scan: bool) -> forward.Forward1D:
        params = forward.Forward1DParameters(
            Lscan=10.0,
            Wdevice=5.0,
            Nscan=100,
            phi=phi,
            gamma=gamma,
            invert_normal=invert_normal,
            invert_scan=invert_scan,
        )
        # Force Ndevice = 50 (even) regardless of gamma, so that
        # Nscan - Ndevice is even and the padding is reversal-symmetric.
        params.Wdevice = 50 * params.dyproj
        return forward.Forward1D(params)

    fwd_00 = make(False, False)
    fwd_01 = make(False, True)
    fwd_10 = make(True, False)
    fwd_11 = make(True, True)

    rng = np.random.default_rng(0)
    current = rng.uniform(-5, 5, size=fwd_00.params.Ndevice)
    # torch.from_numpy rejects negative strides, hence the copies.
    current_rev = current[::-1].copy()

    s00 = fwd_00.forward(current)
    s01 = fwd_01.forward(current)
    s10 = fwd_10.forward(current)
    s11 = fwd_11.forward(current)
    s00_mirror = fwd_00.forward(current_rev)

    assert np.allclose(s10, -s00, rtol=1e-5, atol=1e-5)
    assert np.allclose(s11, -s01, rtol=1e-5, atol=1e-5)
    assert np.allclose(s01[::-1], s00_mirror, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("phi", [0.0, 0.35, -0.6])
def test_invertflags_fullstack_2d(phi: float) -> None:
    """Full-stack contract of the invert flags at arbitrary phi.

    invert_normal:  S -> -S.
    invert_scan:    S_scan(Jx, Jy) == yrev(S_base(yrev(Jx), yrev(Jy))).

    No sign flip on Jy: the vector parity under the mirror is carried by
    kernel1_sign vs kernel2_sign. Exact for any n_alpha (alpha -> -alpha
    maps the quadrature onto itself); the mirror extension, cosine window,
    and center extraction all commute with y-array reversal.
    """

    def make(invert_normal: bool, invert_scan: bool) -> forward.Forward2DCurrent:
        params = forward.Forward2DParameters(
            Lx=2.0,
            Ly=1.0,
            Nx=40,
            Ny=20,
            phi=phi,
            invert_normal=invert_normal,
            invert_scan=invert_scan,
        )
        return forward.Forward2DCurrent(params)

    fwd_00 = make(False, False)
    fwd_01 = make(False, True)
    fwd_10 = make(True, False)
    fwd_11 = make(True, True)

    rng = np.random.default_rng(0)
    current_x = rng.uniform(-5, 5, size=(40, 20))
    current_y = rng.uniform(-5, 5, size=(40, 20))
    current_x_rev = current_x[:, ::-1].copy()
    current_y_rev = current_y[:, ::-1].copy()

    s00 = fwd_00.forward(current_x, current_y)
    s01 = fwd_01.forward(current_x, current_y)
    s10 = fwd_10.forward(current_x, current_y)
    s11 = fwd_11.forward(current_x, current_y)
    s00_mirror = fwd_00.forward(current_x_rev, current_y_rev)

    assert np.allclose(s10, -s00, rtol=1e-6, atol=1e-6)
    assert np.allclose(s11, -s01, rtol=1e-6, atol=1e-6)
    assert np.allclose(s01[:, ::-1], s00_mirror, rtol=1e-6, atol=1e-6)


def test_wdevice_roundig():
    # Test that Wdevice is rounded to an integer number of pixels, and that
    # the resulting Ndevice is even.
    params = forward.Forward1DParameters(
        Lscan=10.0,
        Wdevice=5.04,
        Nscan=100,  # dyproj = Lscan / Nscan = 0.1 for gamma=0.0
        gamma=0.0,
    )

    assert np.isclose(params.Wdevice, 5.0)  # Rounded to nearest pixel
    remainder = params.Wdevice % params.dyproj
    assert np.isclose(remainder, 0.0, rtol=1e-10, atol=1e-10) or np.isclose(
        params.dyproj - remainder, 0.0, rtol=1e-10, atol=1e-10
    )


def test_numpy_torch_conversion():
    params1d = forward.Forward1DParameters(
        Lscan=10.0, Wdevice=5.0, Nscan=100, gamma=0.0
    )
    forward1d = forward.Forward1D(params1d)
    rng = np.random.default_rng(0)
    current1d = rng.uniform(-5, 5, size=params1d.Ndevice)
    signal1d_torch = forward1d.forward_t(torch.from_numpy(current1d))
    signal1d_numpy = forward1d.forward(current1d)
    signal1d_torch_to_numpy = signal1d_torch.detach().cpu().numpy()
    assert np.allclose(signal1d_numpy, signal1d_torch_to_numpy, rtol=1e-6, atol=1e-6)

    params2d = forward.Forward2DParameters(Lx=4.0, Ly=2.0, Nx=100, Ny=50, phi=0.0)
    forward2d = forward.Forward2DCurrent(params2d)
    current2d_x = rng.uniform(-5, 5, size=(params2d.Nx, params2d.Ny))
    current2d_y = rng.uniform(-5, 5, size=(params2d.Nx, params2d.Ny))
    signal2d_torch = forward2d.forward_t(
        torch.from_numpy(current2d_x), torch.from_numpy(current2d_y)
    )
    signal2d_numpy = forward2d.forward(current2d_x, current2d_y)
    signal2d_torch_to_numpy = signal2d_torch.detach().cpu().numpy()
    assert np.allclose(signal2d_numpy, signal2d_torch_to_numpy, rtol=1e-6, atol=1e-6)


def test_cuda_device():
    # Test that setting device to CUDA works and that input/output remain
    # on GPU for both 1D and 2D forward models.

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this system.")

    params1d = forward.Forward1DParameters(
        Lscan=10.0, Wdevice=5.0, Nscan=100, gamma=0.0
    )
    forward1d = forward.Forward1D(params1d, device="cuda")
    rng = np.random.default_rng(0)
    current1d = rng.uniform(-5, 5, size=params1d.Ndevice)
    current1d_torch = torch.from_numpy(current1d).to("cuda")
    signal1d_torch = forward1d.forward_t(current1d_torch)
    assert signal1d_torch.device.type == "cuda"

    params2d = forward.Forward2DParameters(Lx=4.0, Ly=2.0, Nx=100, Ny=50, phi=0.0)
    forward2d = forward.Forward2DCurrent(params2d, device="cuda")
    current2d_x = rng.uniform(-5, 5, size=(params2d.Nx, params2d.Ny))
    current2d_y = rng.uniform(-5, 5, size=(params2d.Nx, params2d.Ny))
    current2d_x_torch = torch.from_numpy(current2d_x).to("cuda")
    current2d_y_torch = torch.from_numpy(current2d_y).to("cuda")
    signal2d_torch = forward2d.forward_t(current2d_x_torch, current2d_y_torch)
    assert signal2d_torch.device.type == "cuda"
