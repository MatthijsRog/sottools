import numpy as np
import torch

import sottools.backward as backward
import sottools.forward as forward


def test_1dinversion(inverter1d_with_random_currents):
    inverter, currents = inverter1d_with_random_currents
    ydevice, recovered_current, loss_history = inverter.fit(lr=1e-2, n_steps=10000)

    assert np.allclose(
        recovered_current, currents, rtol=1e-2, atol=8e-2 * np.max(currents)
    )


def test_oversampled1dinversion(oversampledinverter1d_with_random_currents):
    inverter, currents = oversampledinverter1d_with_random_currents
    ydevice, recovered_current, loss_history = inverter.fit(lr=1e-2, n_steps=10000)
    assert np.allclose(
        recovered_current, currents, rtol=1e-2, atol=8e-2 * np.max(currents)
    )


def test_2dinversion(inverter2d_with_uniform_current):
    inverter, streamfunction = inverter2d_with_uniform_current
    res = inverter.fit(lr=1e-2, n_steps=1000)
    np.allclose(
        res["streamfunction"],
        streamfunction,
        rtol=1e-3,
        atol=1e-3 * np.max(streamfunction),
    )


def test_regularizers_1d():
    rng = np.random.default_rng(0)

    params = forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=100)
    inverter = backward.Inverter1D(forward.Forward1D(params), np.zeros(params.Nscan))

    # Test TV-1 regularization: flat functions are in the null-space
    inverter.lam1 = 1.0
    inverter.lam2 = 0.0
    inverter.lam3 = 0.0
    inverter.lam4 = 0.0
    reg = inverter._regularization(torch.ones(params.Ndevice)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test TV-2 regularization: linear functions AND flat functions are in the
    # null-space
    inverter.lam1 = 0.0
    inverter.lam2 = 1.0
    inverter.lam3 = 0.0
    inverter.lam4 = 0.0
    ydevice = params.ydevice.detach().cpu().numpy()
    linear_func = ydevice
    reg = inverter._regularization(torch.from_numpy(linear_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)
    flat_func = np.ones(params.Ndevice)
    reg = inverter._regularization(torch.from_numpy(flat_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test positivity regularization: all-positive function with much detail
    # should still be in null-space
    inverter.lam1 = 0.0
    inverter.lam2 = 0.0
    inverter.lam3 = 1.0
    inverter.lam4 = 0.0
    detailed_positive_func = rng.uniform(0.0, 1.0, size=params.Ndevice)
    reg = (
        inverter._regularization(torch.from_numpy(detailed_positive_func))
        .detach()
        .cpu()
        .numpy()
    )
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test edge positivity: negative in bulk, positive at edges should be in null-space
    inverter.lam1 = 0.0
    inverter.lam2 = 0.0
    inverter.lam3 = 0.0
    inverter.lam4 = 1.0
    inverter.n_edge = 10
    detailed_func = rng.uniform(-1.0, 1.0, size=params.Ndevice)
    detailed_func[: inverter.n_edge] = rng.uniform(0.0, 1.0, size=inverter.n_edge)
    detailed_func[-inverter.n_edge :] = rng.uniform(0.0, 1.0, size=inverter.n_edge)
    reg = (
        inverter._regularization(torch.from_numpy(detailed_func)).detach().cpu().numpy()
    )
    assert np.isclose(reg, 0.0, atol=1e-5)


def test_regularizers_2d(bar_mesh_london):
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi + 1e-9, np.pi - 1e-9)

    params = forward.Forward2DParameters(Lx=8.0, Ly=8.0, Nx=100, Ny=100, phi=phi)
    mesh = bar_mesh_london
    meshtogrid = forward.MeshToGrid(mesh, params)
    fwd = forward.Forward2DCurrent(params)
    inverter = backward.Inverter2D(meshtogrid, fwd, np.zeros((100, 100)))

    # Test Tikhonov:
    inverter.lam1 = 1.0  # Tikhonov
    inverter.lam2 = 0.0
    y = mesh.vertices[:, 1]
    linear_streamfunc = y - np.min(y)
    reg = (
        inverter._tikhonov_regularization(
            torch.from_numpy(linear_streamfunc).to(dtype=torch.float32)
        )
        .detach()
        .cpu()
        .numpy()
    )
    assert np.isclose(reg, 0.0, atol=1e-1)

    # Test TV on J, again with a linear streamfunction
    inverter.lam1 = 0.0
    inverter.lam2 = 1.0  # TV on J
    reg = (
        inverter._tv_regularization(
            torch.from_numpy(linear_streamfunc).to(dtype=torch.float32)
        )
        .detach()
        .cpu()
        .numpy()
    )
    print(reg)
    assert np.isclose(reg, 0.0, atol=1e-1)

    # --- Scaling tests ---
    y_mid = (np.max(y) + np.min(y)) / 2
    kink_1 = linear_streamfunc + 1.0 * np.maximum(y - y_mid, 0)
    kink_2 = linear_streamfunc + 2.0 * np.maximum(y - y_mid, 0)

    # Tikhonov-2: quadratic in jump height (L2 on ∇J)
    inverter.lam1 = 1.0
    inverter.lam2 = 0.0
    reg_1 = inverter._tikhonov_regularization(
        torch.from_numpy(kink_1).to(dtype=torch.float32)
    ).item()
    reg_2 = inverter._tikhonov_regularization(
        torch.from_numpy(kink_2).to(dtype=torch.float32)
    ).item()
    assert reg_1 > 0
    assert np.isclose(reg_2 / reg_1, 4.0, rtol=0.15)

    # TV on J: linear in jump height (L1 on [J])
    # Set huber_e small so we're firmly in L1 regime
    inverter._huber_e = 1e-8
    inverter.lam1 = 0.0
    inverter.lam2 = 1.0
    reg_1 = inverter._tv_regularization(
        torch.from_numpy(kink_1).to(dtype=torch.float32)
    ).item()
    reg_2 = inverter._tv_regularization(
        torch.from_numpy(kink_2).to(dtype=torch.float32)
    ).item()
    assert reg_1 > 0
    assert np.isclose(reg_2 / reg_1, 2.0, rtol=0.15)

    # --- Smooth vs sharp (same total current change, different distribution) ---
    A = 2.0
    w = 0.5  # transition width
    sharp = linear_streamfunc + A * np.maximum(y - y_mid, 0)
    smooth = linear_streamfunc + A * w * np.log1p(np.exp((y - y_mid) / w))

    # Tikhonov-2 strongly prefers smooth (spreading kink reduces L2 norm)
    inverter.lam1 = 1.0
    inverter.lam2 = 0.0
    reg_sharp = inverter._tikhonov_regularization(
        torch.from_numpy(sharp).to(dtype=torch.float32)
    ).item()
    reg_smooth = inverter._tikhonov_regularization(
        torch.from_numpy(smooth).to(dtype=torch.float32)
    ).item()
    assert reg_smooth < 0.5 * reg_sharp

    # TV on J roughly indifferent (same total variation)
    inverter.lam1 = 0.0
    inverter.lam2 = 1.0
    reg_sharp = inverter._tv_regularization(
        torch.from_numpy(sharp).to(dtype=torch.float32)
    ).item()
    reg_smooth = inverter._tv_regularization(
        torch.from_numpy(smooth).to(dtype=torch.float32)
    ).item()
    assert np.isclose(reg_smooth, reg_sharp, rtol=0.3)


def test_1dreset(inverter1d_with_random_currents):
    inverter, _ = inverter1d_with_random_currents
    _, recovered_current, loss_history = inverter.fit(lr=1e-2, n_steps=2000)
    last_loss = loss_history[-1]
    # Check non-zero currents were recovered
    assert np.max(np.abs(recovered_current)) > 0.0
    _, _, loss_history = inverter.fit(lr=1e-2, n_steps=1)
    # Check we actually continued where we left off
    assert np.isclose(loss_history[0], last_loss, atol=1e-5)
    inverter.reset()
    assert np.allclose(inverter._current.detach().cpu().numpy(), 0.0, atol=1e-9)


def test_callbacks(inverter1d_with_random_currents):
    # Test callback mechanisms
    class RecordingCallback:
        """Minimal FitCallback that records all received states."""

        def __init__(self) -> None:
            self.steps: list[backward.FitState1D] = []
            self.finishes: list[backward.FitState1D] = []

        def on_step(self, state: backward.FitState1D) -> None:
            self.steps.append(state)

        def on_finish(self, state: backward.FitState1D) -> None:
            self.finishes.append(state)

    params = forward.Forward1DParameters(
        Lscan=10.0,
        Wdevice=5,
        Nscan=200,
        rho1=0.2,
        rho2=0.3,
        height=0.4,
        phi=np.deg2rad(60.0),
    )

    inverter, _ = inverter1d_with_random_currents
    cb = RecordingCallback()
    inverter.fit(lr=1e-2, n_steps=2000, callback=cb, callback_every=500)

    # Assert that the callback was called at the expected intervals
    assert len(cb.steps) == 4  # 2000 steps, callback every 500
    assert len(cb.finishes) == 1  # Callback on finish

    # Assert loss values are decreasing
    losses = [state.loss for state in cb.steps]
    assert all(
        earlier >= later for earlier, later in zip(losses, losses[1:], strict=False)
    )

    # Assert that for each step, all expected attributes are present and have
    # reasonable values
    for i, state in enumerate(cb.steps):
        assert isinstance(state.step, int)
        assert isinstance(state.total_steps, int)
        assert isinstance(state.loss, float)
        assert isinstance(state.fidelity, float)
        assert isinstance(state.regularization, float)
        assert isinstance(state.current, np.ndarray)
        assert isinstance(state.predicted_signal, np.ndarray)
        assert isinstance(state.to_fit_signal, np.ndarray)
        assert isinstance(state.xaxis_current, np.ndarray)
        assert isinstance(state.xaxis_signal, np.ndarray)

        assert state.current.shape == (params.Ndevice,)
        assert state.predicted_signal.shape == (params.Nscan,)
        assert state.to_fit_signal.shape == (params.Nscan,)
        assert state.xaxis_current.shape == (params.Ndevice,)
        assert state.xaxis_signal.shape == (params.Nscan,)

        assert state.step == i * 500
        assert state.total_steps == 2000
        assert np.isclose(state.loss, state.fidelity + state.regularization, rtol=1e-5)

        # Make sure all values are reasonable (not NaN or Inf)
        assert np.isfinite(state.loss)
        assert np.isfinite(state.fidelity)
        assert np.isfinite(state.regularization)
        assert np.all(np.isfinite(state.current))

    # %%
