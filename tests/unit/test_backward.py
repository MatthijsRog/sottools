import numpy as np
import pytest
import scipy.optimize
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
    res = inverter.fit(lr=1, n_steps=100)
    assert np.allclose(
        res["streamfunction"],
        streamfunction,
        rtol=1e-3,
        atol=1e-3 * np.max(streamfunction),
    )


def test_regularizers_1d():
    rng = np.random.default_rng(0)

    params = forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=100)

    # Test TV-1 regularization: flat functions are in the null-space
    regularizer = backward.TV1()
    reg = regularizer(torch.ones(params.Ndevice)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test TV-2 regularization: linear functions AND flat functions are in the
    # null-space
    regularizer = backward.TV2()
    ydevice = params.ydevice.detach().cpu().numpy()
    linear_func = ydevice
    reg = regularizer(torch.from_numpy(linear_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)
    flat_func = np.ones(params.Ndevice)
    reg = regularizer(torch.from_numpy(flat_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test positivity regularization: all-positive function with much detail
    # should still be in null-space
    regularizer = backward.Positivity()
    detailed_positive_func = rng.uniform(0.0, 1.0, size=params.Ndevice)
    reg = regularizer(torch.from_numpy(detailed_positive_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test edge positivity: negative in bulk, positive at edges should be in null-space
    regularizer = backward.EdgeRingingSuppression(n_edge=10)
    detailed_func = rng.uniform(-1.0, 1.0, size=params.Ndevice)
    detailed_func[: regularizer.n_edge] = rng.uniform(0.0, 1.0, size=regularizer.n_edge)
    detailed_func[-regularizer.n_edge :] = rng.uniform(
        0.0, 1.0, size=regularizer.n_edge
    )
    reg = regularizer(torch.from_numpy(detailed_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test Tikhonov-1 (smoothness, 1st derivatives)
    regularizer = backward.Tikhonov1()
    flat_func = np.ones(params.Ndevice)
    reg = regularizer(torch.from_numpy(flat_func)).detach().cpu().numpy()
    assert np.isclose(reg, 0.0, atol=1e-5)

    # Test Tikhonov-2 (smoothness, 2nd derivatives)
    regularizer = backward.Tikhonov2()
    linear_func = ydevice
    reg = regularizer(torch.from_numpy(linear_func)).detach().cpu().numpy()
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


# ----------------------
# Coinversion Unit Tests
# ----------------------


def test_folds():
    coinverter_k1 = backward.Coinversion1D(
        forward.Forward1D(
            forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=200)
        ),
        np.zeros(200),
        backward.Tikhonov1(),
        crossvalidation_size=1,
    )

    # Test that there is only 1 training and 1 test set and these are both purely True
    assert len(coinverter_k1._folds) == 1
    train_mask, test_mask = (m.cpu().numpy() for m in coinverter_k1._folds[0])
    assert len(train_mask) == 200
    assert len(test_mask) == 200
    assert np.all(train_mask)
    assert np.all(test_mask)

    coinverter_k3 = backward.Coinversion1D(
        forward.Forward1D(
            forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=200)
        ),
        np.zeros(200),
        backward.Tikhonov1(),
        crossvalidation_size=3,
    )

    # Test that there are 3 training sets / 3 test sets
    # + that the test sets are disjoint and cover all indices
    # + that the training sets are the inverse of the test sets
    assert len(coinverter_k3._folds) == 3
    test_indices = []
    for train_mask, test_mask in coinverter_k3._folds:
        train_mask, test_mask = train_mask.cpu().numpy(), test_mask.cpu().numpy()
        assert len(train_mask) == 200
        assert len(test_mask) == 200
        test_indices.append(np.where(test_mask)[0])
        # Check that training and test masks are disjoint
        assert not np.any(train_mask & test_mask)
        # Check that training mask is the inverse of the test mask
        assert np.all(train_mask == ~test_mask)
    # Check that test indices are disjoint and cover all indices
    all_test_indices = np.concatenate(test_indices)
    assert len(np.unique(all_test_indices)) == 200


def test_constructor():
    # Making a coinverter with height < rho2 * sin(phi) should raise a ValueError
    params = forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=200, rho2=0.5)
    fwd = forward.Forward1D(params)
    signal = fwd.forward(np.zeros(params.Ndevice))
    regularizer = backward.Tikhonov1()
    with pytest.raises(ValueError):
        _ = backward.Coinversion1D(
            fwd,
            signal,
            regularizer,
            initial_height=0.1,
            initial_phi_eff=np.deg2rad(80.0),
        )

    coinverter = backward.Coinversion1D(
        fwd, signal, regularizer, initial_height=0.9, initial_phi_eff=np.deg2rad(80.0)
    )

    # Assert that sensor_parameters at this point is equal to the initial parameters
    h, rho1, rho2, phi_eff = coinverter.sensor_parameters
    assert np.isclose(h, 0.9)
    assert np.isclose(rho1, params.rho1)
    assert np.isclose(rho2, params.rho2)
    assert np.isclose(phi_eff, np.deg2rad(80.0))

    # Assert that at this stage, prediction raises error
    with pytest.raises(ValueError):
        _ = coinverter.prediction


def test_coinverter_set_guards():
    class lrs(backward.LearningRateScheduler):
        def __init__(self, val):
            self._val = val

        def __call__(self, epoch):
            return self._val

    class rs(backward.RegularizerScheduler):
        def __init__(self, val):
            self._val = val

        def __call__(self, epoch):
            return self._val * backward.Tikhonov1()

    coinverter = backward.Coinversion1D(
        forward.Forward1D(
            forward.Forward1DParameters(Lscan=10.0, Wdevice=5, Nscan=200)
        ),
        np.zeros(200),
        lr_sensor=lrs(0.1),
        lr_current=lrs(0.1),
        regularizer=rs(0.1),
    )

    # Assert that setting a learning rate to a float now raises a ValueError
    with pytest.raises(ValueError):
        coinverter.lr_sensor = 0.1
    with pytest.raises(ValueError):
        coinverter.lr_current = 0.1
    # Same for regularizer:
    with pytest.raises(ValueError):
        coinverter.regularizer = backward.Tikhonov1()

    # Check that setting to ANOTHER scheduler does NOT raise an error
    coinverter.lr_sensor = lrs(0.2)
    coinverter.lr_current = lrs(0.2)
    coinverter.regularizer = rs(0.2)


def test_coinversion_convergence_checker():
    # Random seed
    rng = np.random.default_rng(0)

    conv = backward.CoinversionConvergenceChecker(
        objtol=1e-7,
        fidtol=1e-7,
        heighttol=1e-4,
        phitol=1e-4,
        nwindow=10,
        nextrapolate=100,
    )

    # Make artificial history  and check
    # (1) passing a short history returns False
    history = backward.CoinversionFitHistory()
    for i in range(5):
        history.append(
            fidelity=1.0 / (i + 1),
            regularization_terms={"term": 0.0},
            regularization=0.0,
            objective=0.0,
            height=1.0,
            phi=np.deg2rad(45.0),
        )
    assert not conv.check(history)

    # (2) passing a history that is constant with no noise returns True
    history = backward.CoinversionFitHistory()
    for _ in range(15):
        history.append(
            fidelity=1.0,
            regularization_terms={"term": 0.0},
            regularization=0.0,
            objective=0.0,
            height=1.0,
            phi=np.deg2rad(45.0),
        )
    assert conv.check(history)

    # (3) passing a history that is constant but with a large noise returns False
    history = backward.CoinversionFitHistory()
    for _ in range(15):
        history.append(
            fidelity=1.0 + 1e-3 * rng.standard_normal(),
            regularization_terms={"term": 0.0},
            regularization=0.0,
            objective=0.0,
            height=1.0 + 1e-3 * rng.standard_normal(),
            phi=np.deg2rad(45.0) + 1e-3 * rng.standard_normal(),
        )
    assert not conv.check(history)

    # (4) passing a history that is decreasing with no noise
    # And by hand guaranteed last 10 steps within tol
    # However extrapolates out of the window should return False
    history = backward.CoinversionFitHistory()
    for i in range(15):
        history.append(
            fidelity=1.0,
            regularization_terms={"term": 0.0},
            regularization=0.0,
            objective=0.0,
            height=1.0 + 5e-6 * i,
            phi=np.deg2rad(45.0),
        )
    assert not conv.check(history)


@pytest.mark.parametrize("invert_scan", [False, True])
def test_cached_vs_explicit_kernels(invert_scan):
    # Forward model whose stored height/phi deliberately differ from the coinverter's
    # starting point, so a stale or sign-flipped cache would be detected.
    params = forward.Forward1DParameters(
        Lscan=10.0,
        Wdevice=5,
        Nscan=200,
        rho1=0.2,
        rho2=0.3,
        height=0.4,
        phi=np.deg2rad(60.0),
        invert_scan=invert_scan,
    )
    fwd = forward.Forward1D(params)
    coinverter = backward.Coinversion1D(
        fwd,
        np.zeros(params.Nscan),
        backward.Tikhonov1(),
        initial_height=0.55,
        initial_phi_eff=np.deg2rad(-50.0),
    )

    rng = np.random.default_rng(1)
    current = torch.from_numpy(rng.standard_normal(params.Ndevice)).to(backward.DTYPE)
    sp = coinverter._full_sensor_parameters(coinverter._sensor_parameters_free.detach())

    with torch.no_grad():
        # Explicit path: kernel computed on the fly from (h, rho1, rho2, phi_eff)
        explicit = coinverter._forward.forward_t(current, sp)
        # Signal with the original, not-yet-updated cached kernel
        stale = coinverter._forward.forward_t(current, None)
        # Cached path: write the sensor parameters into forward.params, then reuse
        coinverter._cache_kernels_into_forward(
            coinverter._sensor_parameters_free, coinverter._forward
        )
        cached = coinverter._forward.forward_t(current, None)

    scale = explicit.abs().max()
    # Sanity: the test is only meaningful if the stale kernel differs noticeably
    assert not torch.allclose(stale, explicit, rtol=0, atol=1e-2 * scale)
    # The actual check: cached and explicit kernels give the same signal
    assert torch.allclose(cached, explicit, rtol=0, atol=1e-5 * scale)


# -----------------------------------------
# Coinversion End-To-End real fitting tests
# -----------------------------------------


@pytest.mark.parametrize("K", [1, 2])
@pytest.mark.parametrize("invert_scan", [False, True])
def test_coinv_recovery(K, invert_scan, request, make_coinverter):
    problem = request.getfixturevalue(
        "coinv_problem_inverted" if invert_scan else "coinv_problem"
    )
    _, data, current_true, params, noise_sigma = problem
    coinverter = make_coinverter(problem, crossvalidation_size=K)

    # Run the inverter and assert that
    # height comes out within 0.03 um of the true value
    # phi_eff comes out within 1 degree of the true value
    # rms current reconstruction error within 8% of the maximum current
    res = coinverter.fit(n_steps_sensor=300, n_steps_current=200)
    height, _, _, phi_eff = res["final_sensor_parameters"]

    # Sensor parameters (inverted geometry has a larger, genuine height bias)
    h_tol = 0.06 if invert_scan else 0.03
    assert abs(height - params.height) < h_tol
    assert abs(phi_eff - params.phi_eff) < np.deg2rad(1.0)

    # Current: rms error relative to the peak current
    rms = np.sqrt(np.mean((coinverter.current - current_true) ** 2))
    assert rms < 0.08 * np.max(current_true)

    # Residual: neither overfitting (chi2 << 1) nor a stalled/broken fit (chi2 >> 1)
    chi2 = np.mean((coinverter.prediction - data) ** 2) / noise_sigma**2
    assert 1.0 < chi2 < 2.0


def test_coinv_contract(request, make_coinverter):
    # Verifies that callbacks, histories, and resets are implemented consistently
    class Testcallback:
        # Callback that keeps a record, for testing
        def __init__(self):
            self.fitstates = []
            self.called_on_step = []
            self.called_on_finish = 0

        def on_step(self, state):
            self.fitstates.append(state)
            self.called_on_step.append(state.step)

            # Assert that objective ~ fidelity + regularization (only holds at K=1)
            assert np.isclose(
                state.loss, state.fidelity + state.regularization, rtol=1e-5
            )

        def on_finish(self, state):
            self.called_on_finish += 1

    class Testscheduler(backward.RegularizerScheduler):
        # Regularizer scheduler that returns a constant regularizer and
        # keeps a record meant for testing
        def __init__(self, reg):
            self.reg = reg
            self.called_on_step = []

        def __call__(self, step):
            self.called_on_step.append(step)
            return self.reg

    cb = Testcallback()
    ts = Testscheduler(1e-3 * backward.Tikhonov1())
    problem = request.getfixturevalue("coinv_problem")
    coinverter = make_coinverter(
        problem, crossvalidation_size=1, callback=cb, callback_every=2, regularizer=ts
    )

    coinverter.fit(n_steps_sensor=4)
    sensor_params_4steps = np.copy(coinverter.sensor_parameters)
    res = coinverter.fit(n_steps_sensor=4)
    assert res["steps_total"] == 8
    assert cb.called_on_finish == 2
    assert cb.called_on_step == [1, 3, 5, 7]
    assert set(range(8)) <= set(coinverter.regularizer.called_on_step)

    no_cache_coinverter = make_coinverter(
        problem,
        crossvalidation_size=1,
        callback=None,
        callback_every=2,
        regularizer=1e-3 * backward.Tikhonov1(),
        cache_kernels=False,
    )
    _ = no_cache_coinverter.fit(n_steps_sensor=8)

    cached_height, _, _, cached_phi_eff = coinverter.sensor_parameters
    no_cache_height, _, _, no_cache_phi_eff = no_cache_coinverter.sensor_parameters
    assert np.isclose(cached_height, no_cache_height, atol=1e-6)
    assert np.isclose(cached_phi_eff, no_cache_phi_eff, atol=1e-6)

    coinverter.reset()
    assert coinverter.current is None
    assert len(coinverter.history.height_history) == 0
    coinverter.fit(n_steps_sensor=4)
    assert len(coinverter.history.height_history) == 4
    assert np.allclose(coinverter.sensor_parameters, sensor_params_4steps, atol=1e-6)


def test_coinv_convergence(coinv_problem, make_coinverter):
    # Lax checker: objective/fidelity criteria disabled, height/phi must settle
    checker = backward.CoinversionConvergenceChecker(
        objtol=np.inf,
        fidtol=np.inf,
        heighttol=5e-3,
        phitol=5e-3,
        nwindow=10,
        nextrapolate=None,
    )
    params = coinv_problem[3]
    coinverter = make_coinverter(coinv_problem, convergence_checker=checker)
    res = coinverter.fit(n_steps_sensor=300, n_steps_current=200)

    assert res["message"] == "converged"
    assert res["steps"] < 300
    assert abs(coinverter.sensor_parameters[0] - params.height) < 0.03


# -----------------------------
# Coinversion2D End-To-End tests
# -----------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "K, invert_scan, lumped_mass",
    [
        (1, False, True),
        (2, False, True),
        (1, True, True),
        (2, True, True),
        (1, False, False),
    ],
)
def test_coinv2d_recovery(K, invert_scan, lumped_mass, request, make_coinverter2d):
    problem = request.getfixturevalue(
        "coinv2d_problem_inverted" if invert_scan else "coinv2d_problem"
    )
    coinverter = make_coinverter2d(
        problem, crossvalidation_size=K, lumped_mass=lumped_mass
    )
    res = coinverter.fit(n_steps_sensor=500, n_steps_current=500)
    h, _, _, phi_eff = res["final_sensor_parameters"]

    assert abs(h - problem.params.height) < 0.01
    assert abs(phi_eff - problem.params.phi_eff) < np.deg2rad(0.5)

    I_total = problem.I_total_true
    assert np.isclose(coinverter.solution[0], I_total, rtol=0.02)
    error = coinverter.fitted_streamfunction - problem.streamfunction_true
    assert np.sqrt(np.mean(error**2)) < 0.05 * I_total

    n_data = problem.data.size
    chi2 = np.mean((coinverter.prediction - problem.data) ** 2) / problem.noise_sigma**2
    chi2_min = 1.0 - coinverter.solution.size / n_data - 3 * np.sqrt(2 / n_data)
    assert chi2_min < chi2 < 1.05


@pytest.mark.slow
def test_coinv2d_recovery_mismatch(coinv2d_problem_mismatch, make_coinverter2d):
    problem = coinv2d_problem_mismatch
    res = make_coinverter2d(problem).fit(n_steps_sensor=500, n_steps_current=500)
    h, _, _, phi_eff = res["final_sensor_parameters"]

    assert abs(h - problem.params.height) < 0.01
    assert abs(phi_eff - problem.params.phi_eff) < np.deg2rad(0.5)


def test_coinv2d_contract(coinv2d_problem, make_coinverter2d, capsys, single_thread):
    # Note: need to run on single thread to ensure deterministic output before/after
    # reset.
    class RecordingCallback:
        def __init__(self):
            self.steps = []
            self.finishes = 0

        def on_step(self, state):
            self.steps.append(state)

        def on_finish(self, state):
            self.finishes += 1

    def n_iterations_printed():
        out = capsys.readouterr().out
        return sum(line.startswith("Iteration ") for line in out.splitlines())

    problem = coinv2d_problem
    n_vertices = len(problem.meshtogrid.mesh.vertices)
    cb = RecordingCallback()
    coinverter = make_coinverter2d(
        problem, crossvalidation_size=1, callback=cb, callback_every=1, verbose=True
    )

    res1 = coinverter.fit(n_steps_sensor=4)
    n_iter1 = n_iterations_printed()
    params_first = np.copy(coinverter.sensor_parameters)
    res2 = coinverter.fit(n_steps_sensor=4)
    n_iter2 = n_iterations_printed()

    # Callbacks
    assert len(cb.steps) == n_iter1 + n_iter2
    assert cb.finishes == 2
    steps = [state.step for state in cb.steps]
    assert all(a < b for a, b in zip(steps, steps[1:], strict=False))
    for state in cb.steps:
        assert np.isclose(state.loss, state.fidelity + state.regularization, rtol=1e-5)
        assert state.streamfunction.shape == (n_vertices,)
        assert state.predicted_signal.shape == problem.data.shape

    # Continuation: second fit starts where the first ended
    assert res2["steps_total"] == res1["steps"] + res2["steps"]
    assert np.isclose(
        coinverter.history.height_history[res1["steps"]], params_first[0], atol=1e-6
    )

    # Reset
    coinverter.reset()
    assert coinverter.solution is None
    assert coinverter.fitted_streamfunction is None
    assert len(coinverter.history.height_history) == 0
    coinverter.fit(n_steps_sensor=4)
    assert np.allclose(coinverter.sensor_parameters, params_first, atol=1e-6)

    # Cached and uncached kernels give the same fit
    no_cache = make_coinverter2d(problem, crossvalidation_size=1, cache_kernels=False)
    no_cache.fit(n_steps_sensor=4)
    assert np.allclose(no_cache.sensor_parameters, params_first, atol=1e-5)


def test_coinv2d_convergence_checker(coinv2d_problem, make_coinverter2d):
    checker = backward.CoinversionConvergenceChecker(
        objtol=np.inf,
        fidtol=np.inf,
        heighttol=np.inf,
        phitol=np.inf,
        nwindow=3,
        nextrapolate=None,
    )
    coinverter = make_coinverter2d(
        coinv2d_problem, crossvalidation_size=1, convergence_checker=checker
    )
    res = coinverter.fit()

    assert res["message"] == "converged"
    assert res["steps"] < 10
    assert "StopIteration" in res["optimizer_message"]


@pytest.mark.slow
def test_coinv2d_initial_streamfunction(coinv2d_problem, make_coinverter2d):
    problem = coinv2d_problem
    zero_start = make_coinverter2d(problem, crossvalidation_size=1)
    true_start = make_coinverter2d(
        problem,
        crossvalidation_size=1,
        initial_streamfunction=problem.streamfunction_true,
    )

    h_zero, _, _, phi_zero = zero_start.sensor_parameters
    h_true, _, _, phi_true = true_start.sensor_parameters
    assert abs(h_true - h_zero) < 0.01
    assert abs(phi_true - phi_zero) < np.deg2rad(0.5)


# --------------------------
# Coinversion2D unit tests
# --------------------------


@pytest.mark.parametrize("invert_scan", [False, True])
def test_cached_vs_explicit_kernels_2d(invert_scan, request, make_coinverter2d):
    problem = request.getfixturevalue(
        "coinv2d_problem_inverted" if invert_scan else "coinv2d_problem"
    )
    coinverter = make_coinverter2d(
        problem, initial_height=0.6, initial_phi_eff=np.deg2rad(-50.0)
    )

    rng = np.random.default_rng(1)
    n_vertices = len(problem.meshtogrid.mesh.vertices)
    sv = torch.from_numpy(rng.standard_normal(n_vertices)).to(backward.DTYPE)
    sp = coinverter._full_sensor_parameters(coinverter._sensor_parameters_free.detach())

    with torch.no_grad():
        explicit = coinverter._signal(sv, sp, use_cached_kernels=False)
        stale = coinverter._signal(sv, sp, use_cached_kernels=True)
        coinverter._cache_kernels_into_forward(
            coinverter._sensor_parameters_free, coinverter._currentforward
        )
        cached = coinverter._signal(sv, sp, use_cached_kernels=True)

    scale = explicit.abs().max()
    assert not torch.allclose(stale, explicit, rtol=0, atol=1e-2 * scale)
    assert torch.allclose(cached, explicit, rtol=0, atol=1e-5 * scale)


@pytest.mark.parametrize("use_cg", [True, False])
@pytest.mark.parametrize("K", [1, 2])
def test_coinv2d_hypergradient_fd(
    use_cg, K, coinv2d_problem, make_coinverter2d, monkeypatch
):
    captured = {}

    def stub_minimize(fun, x0, **kwargs):
        captured["fun"] = fun
        return scipy.optimize.OptimizeResult(x=x0, success=True, message="stub")

    monkeypatch.setattr(backward.scipy.optimize, "minimize", stub_minimize)
    coinverter = make_coinverter2d(
        coinv2d_problem, crossvalidation_size=K, use_cg=use_cg
    )
    coinverter.fit(n_steps_sensor=1, n_steps_current=500)
    theta0 = (
        coinverter._sensor_parameters_free.detach().cpu().numpy().astype(np.float64)
    )

    def evaluate(theta):
        coinverter.reset()  # fresh inner solves, so the score is path-independent
        return captured["fun"](theta)

    _, grad = evaluate(theta0)
    delta = 1e-2
    fd = np.array(
        [
            (evaluate(theta0 + delta * e)[0] - evaluate(theta0 - delta * e)[0])
            / (2 * delta)
            for e in np.eye(2)
        ]
    )
    assert np.allclose(grad, fd, rtol=0.1, atol=1e-2 * np.max(np.abs(fd)))


def test_folds_2d(coinv2d_problem, make_coinverter2d):
    ((train, test),) = make_coinverter2d(coinv2d_problem, crossvalidation_size=1)._folds
    assert train.all()
    assert test.all()

    folds = make_coinverter2d(coinv2d_problem, crossvalidation_size=2)._folds
    assert len(folds) == 2
    for train, test in folds:
        assert torch.equal(train, ~test)
        assert not (test[1:, :] & test[:-1, :]).any()
        assert not (test[:, 1:] & test[:, :-1]).any()
    assert (folds[0][1] ^ folds[1][1]).all()


def test_constructor_2d(coinv2d_problem, make_coinverter2d):
    params = coinv2d_problem.params
    with pytest.raises(ValueError):
        make_coinverter2d(
            coinv2d_problem, initial_height=0.1, initial_phi_eff=np.deg2rad(80.0)
        )

    coinverter = make_coinverter2d(
        coinv2d_problem, initial_height=0.9, initial_phi_eff=np.deg2rad(80.0)
    )
    h, rho1, rho2, phi_eff = coinverter.sensor_parameters
    assert np.isclose(h, 0.9)
    assert np.isclose(rho1, params.rho1)
    assert np.isclose(rho2, params.rho2)
    assert np.isclose(phi_eff, np.deg2rad(80.0))
    assert coinverter.solution is None
    assert coinverter.fitted_streamfunction is None
    with pytest.raises(ValueError):
        _ = coinverter.prediction


@pytest.mark.parametrize("cls", [backward.Inverter2D, backward.Coinversion2D])
def test_initial_streamfunction_validation(cls, coinv2d_problem):
    problem = coinv2d_problem
    mesh = problem.meshtogrid.mesh
    good = problem.streamfunction_true

    bad_left = good.copy()
    bad_left[mesh.dirichletleft_mask] = 1.0
    bad_right = good.copy()
    bad_right[np.flatnonzero(mesh.dirichletright_mask)[0]] += 1.0

    for bad in (good[:-1], bad_left, bad_right):
        with pytest.raises(ValueError):
            cls(
                problem.meshtogrid,
                problem.currentforward,
                problem.data,
                initial_streamfunction=bad,
            )

    obj = cls(
        problem.meshtogrid,
        problem.currentforward,
        problem.data,
        initial_streamfunction=good,
    )
    if cls is backward.Inverter2D:
        sv = obj._solution_to_streamvector()
    else:
        sv = obj._solution_to_streamvector(obj._solutions[0])
    assert np.allclose(sv.detach().cpu().numpy(), good, atol=1e-5)


@pytest.mark.parametrize("lumped_mass", [True, False])
def test_regularizers_match_inverter2d(lumped_mass, coinv2d_problem):
    p = coinv2d_problem
    inverter = backward.Inverter2D(
        p.meshtogrid, p.currentforward, p.data, lumped_mass=lumped_mass
    )
    coinverter = backward.Coinversion2D(
        p.meshtogrid, p.currentforward, p.data, lumped_mass=lumped_mass
    )

    rng = np.random.default_rng(0)
    sv = torch.from_numpy(rng.standard_normal(len(p.meshtogrid.mesh.vertices))).to(
        backward.DTYPE
    )
    for name in (
        "_tikhonov_regularization",
        "_tv_regularization",
        "_vorticity_regularization",
    ):
        assert torch.isclose(
            getattr(inverter, name)(sv), getattr(coinverter, name)(sv), rtol=1e-6
        )
