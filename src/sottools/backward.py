"""Inverters to calculate current distributions from SOT signals.

This module contains various classes to parametrize, accelerate and regularize
the inversion of SOT signals to current distributions. Two key points set the
approach of this module apart from other SOT inversion schemes:

1. The inversion is done in real space using iterative optimization of a
regularized loss function.
2. When working in 2D, the inversion is done onto a mesh rather than a grid.
This makes it much easier to incorporate the shape of the sample into the
inversion.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import sys
import time
import typing
import warnings
from typing import Any, Protocol

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import torch

from sottools.forward import Forward1D, Forward2DCurrent, MeshToGrid

type _Fig = matplotlib.figure.Figure
type _Ax = matplotlib.axes.Axes

DTYPE = torch.float32


@dataclasses.dataclass
class FitState1D:
    """State of the 1D fitting process at one particular point.

    Can be used to communicate intermediate results to a user through a
    callback function.

    Parameters
    ----------
    step : int
        Current step number in the fitting process.
    total_steps : int
        Total number of steps in the fitting process.
    loss : float
        Current value of the loss function.
    fidelity : float
        Current value of the fidelity term in the loss function.
    regularization : float
        Current value of the regularization term in the loss function.
    regularization_terms : dict[str, float]
        Dictionary of the individual regularization terms and their values.
    current : np.ndarray
        Current array of size (Ndevice,) representing the current density on
        the device.
    predicted_signal : np.ndarray
        Predicted signal array of size (Nscan,) representing the predicted
        signal based on the current density.
    to_fit_signal : np.ndarray
        Signal array of size (Nscan,) representing the signal to fit.
    xaxis_current : np.ndarray
        X-axis array of size (Ndevice,) representing the positions of the
        current density.
    xaxis_signal : np.ndarray
        X-axis array of size (Nscan,) representing the positions of the signal.
    sensor_parameters : np.ndarray, optional
        Sensor parameters used in the forward model, if applicable. Useful only
        when performing coinversion. Default is None.
    """

    step: int
    total_steps: int
    loss: float
    fidelity: float
    regularization: float
    regularization_terms: dict[str, float]
    current: np.ndarray | None
    predicted_signal: np.ndarray | None
    to_fit_signal: np.ndarray | None
    xaxis_current: np.ndarray | None
    xaxis_signal: np.ndarray | None
    sensor_parameters: np.ndarray | None = None


class FitCallback(Protocol):
    """Protocol for callback functions.

    This protocol generalizes to both 1D and 2D inverters.
    """

    def on_step(self, state: FitState1D) -> None:
        """Call at each (callback) step of the fit."""

    def on_finish(self, state: FitState1D) -> None:
        """Call at the end of the fit."""


class PrintCallback1D:
    """In-shell printed callback for 1D inverters.

    Displays a progress bar and the current loss, fidelity, and regularization
    for a 1D current fit.

    Parameters
    ----------
    bar_width : int
        Width of the progress bar in characters.
    """

    def __init__(self, bar_width: int = 30):
        self._bar_width = bar_width

    def on_step(self, state: FitState1D) -> None:
        """Call at every (callback) step of a 1D fit.

        Parameters
        ----------
        state : FitState1D
            State of the 1D fitting process at this step.
        """
        frac = state.step / state.total_steps
        filled = int(self._bar_width * frac)
        bar = "█" * filled + "░" * (self._bar_width - filled)
        line = (
            f"\r[{bar}] {state.step:5d}/{state.total_steps}  "
            f"loss={state.loss:.3e}  fid={state.fidelity:.3e}  "
            f"reg={state.regularization:.3e}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_finish(self, state: FitState1D) -> None:
        """Call after the last step of a 1D fit.

        Parameters
        ----------
        state : FitState1D
            State of the 1D fitting process at this step.
        """
        self.on_step(state)
        sys.stdout.write("\n")
        sys.stdout.flush()


class JupyterCallback:
    """Jupyter notebook callback for 1D inverters.

    Displays the current, predicted signal, and loss as plots underneath a
    notebook cell.
    """

    def __init__(self) -> None:
        self._losses: list[tuple[int, float]] = []
        self._fig: _Fig | None = None
        self._axes: tuple[_Ax, _Ax, _Ax] | None = None

    def on_step(self, state: FitState1D) -> None:
        """Call at every (callback) step of a 1D fit.

        Parameters
        ----------
        state : FitState1D
            State of the 1D fitting process at this step.
        """
        import IPython.display

        self._losses.append((state.step, state.loss))
        IPython.display.clear_output(wait=True)  # type: ignore[no-untyped-call,unused-ignore]

        if self._fig is None:
            fig, axarr = plt.subplots(3, 1, figsize=(5, 11))
            assert isinstance(axarr, np.ndarray)
            self._fig = fig
            self._axes = (axarr[0], axarr[1], axarr[2])

        assert self._axes is not None
        ax_cur, ax_fit, ax_loss = self._axes
        for ax in self._axes:
            ax.clear()

        if state.current is not None and state.xaxis_current is not None:
            ax_cur.plot(state.xaxis_current, state.current)

        if state.predicted_signal is not None and state.xaxis_signal is not None:
            ax_fit.plot(state.xaxis_signal, state.predicted_signal)
        if state.to_fit_signal is not None and state.xaxis_signal is not None:
            ax_fit.plot(state.xaxis_signal, state.to_fit_signal, ".", ms=2, alpha=0.5)

        ax_cur.set_xlabel("$y''$ (um)")
        ax_cur.set_ylabel("$j(y'')$ (a.u.)")

        ax_fit.set_xlabel("$y$ (um)")
        ax_fit.set_ylabel("$S(y)$ (a.u.)")

        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")

        steps, losses = zip(*self._losses, strict=True)
        ax_loss.semilogy(steps, losses)

        self._fig.tight_layout()
        IPython.display.display(self._fig)  # type: ignore[no-untyped-call,unused-ignore]

    def on_finish(self, state: FitState1D) -> None:
        """Call after the last step of a 1D fit.

        Parameters
        ----------
        state : FitState1D
            State of the 1D fitting process at this step.
        """
        self.on_step(state)
        plt.close(self._fig)


class Regularizer1D(abc.ABC):
    """Abstract base class for 1D regularizers.

    Regularizers can be composed and weighed using standard addition and multiplication.
    """

    __array_ufunc__ = None

    @abc.abstractmethod
    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the regularization term for a given current distribution.

        Parameters
        ----------
        current : torch.Tensor
            Current distribution of shape (Ndevice,).

        Returns
        -------
        torch.Tensor
            Regularization term as a scalar tensor.
        """
        pass

    def __add__(self, other: Regularizer1D) -> Regularizer1D:
        """Compose two regularizers by addition."""
        return SummedRegularizer1D([self, other])

    def __rmul__(self, other: float) -> Regularizer1D:
        """Scale a regularizer by a scalar weight."""
        return ScaledRegularizer1D(self, other)

    def terms(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return regularizer broken down into its constituent terms.

        Parameters
        ----------
        current : torch.Tensor
            Current distribution of shape (Ndevice,).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping term names to their corresponding regularization values.
        """
        return {repr(self): self(current)}


class ScaledRegularizer1D(Regularizer1D):
    """Scaled regularizer that multiplies a regularizer by a scalar weight."""

    def __init__(self, regularizer: Regularizer1D, weight: float):
        self.regularizer = regularizer
        self.weight = weight

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the scaled regularization term for a given current distribution."""
        return self.weight * self.regularizer(current)

    def __repr__(self) -> str:
        """Return a string representation of the scaled regularizer."""
        return f"{self.weight:g}*{self.regularizer!r}"


class SummedRegularizer1D(Regularizer1D):
    """Summed regularizer that adds multiple regularizers together."""

    def __init__(self, children: list[Regularizer1D]):
        self.children = children

        # Flatten any nested SummedRegularizer1D instances
        flattened_children: list[Regularizer1D] = []
        for child in self.children:
            if isinstance(child, SummedRegularizer1D):
                flattened_children.extend(child.children)
            else:
                flattened_children.append(child)
        self.children = flattened_children

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the summed regularization term for a given current distribution."""
        return sum(
            (child(current) for child in self.children),
            start=torch.tensor(0.0, device=current.device, dtype=current.dtype),
        )

    def __repr__(self) -> str:
        """Return a string representation of the summed regularizer."""
        return " + ".join(repr(child) for child in self.children)

    def terms(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return regularizer broken down into its constituent terms.

        Parameters
        ----------
        current : torch.Tensor
            Current distribution of shape (Ndevice,).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping term names to their corresponding regularization values.
        """
        result: dict[str, torch.Tensor] = {}
        for child in self.children:
            for name, value in child.terms(current).items():
                result[name] = result.get(name, 0) + value
        return result


@dataclasses.dataclass(frozen=True)
class TV1(Regularizer1D):
    """1st order Total Variation, Charbonnier-smoothed."""

    eps: float = 1e-3

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the 1st order Total Variation regularization term."""
        d = current[1:] - current[:-1]
        return torch.sum(torch.sqrt(d**2 + self.eps**2) - self.eps)


@dataclasses.dataclass(frozen=True)
class TV2(Regularizer1D):
    """2nd order Total Variation, Charbonnier-smoothed."""

    eps: float = 1e-3

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the 2nd order Total Variation regularization term."""
        d = current[2:] - 2 * current[1:-1] + current[:-2]
        return torch.sum(torch.sqrt(d**2 + self.eps**2) - self.eps)


@dataclasses.dataclass(frozen=True)
class Tikhonov1(Regularizer1D):
    """1st order Tikhonov regularizer: minimizes 2-norm of the derivative."""

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the 1st order Tikhonov regularization term."""
        return torch.sum((current[1:] - current[:-1]) ** 2)


@dataclasses.dataclass(frozen=True)
class Tikhonov2(Regularizer1D):
    """2nd order Tikhonov regularizer: minimizes 2-norm of the second derivative."""

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the 2nd order Tikhonov regularization term."""
        return torch.sum((current[2:] - 2 * current[1:-1] + current[:-2]) ** 2)


@dataclasses.dataclass(frozen=True)
class Positivity(Regularizer1D):
    """Positivity regularizer."""

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the positivity regularization term."""
        return torch.sum(torch.relu(-current) ** 2)


@dataclasses.dataclass(frozen=True)
class EdgeRingingSuppression(Regularizer1D):
    """Edge ringing suppression regularizer.

    Parameters
    ----------
    n_edge: int
        Number of edge elements to consider for suppression. Must be >= 1.
    """

    n_edge: int

    def __post_init__(self) -> None:
        if self.n_edge < 1:
            raise ValueError("n_edge must be >= 1")

    def __call__(self, current: torch.Tensor) -> torch.Tensor:
        """Calculate the edge ringing suppression regularization term."""
        edges = torch.cat([current[: self.n_edge], current[-self.n_edge :]])
        return torch.sum(torch.relu(-edges) ** 2)


class Inverter1D:
    """Inverts a 1D SOT signal to find the corresponding current distribution.

    Uses the Forward1D model in forward.py to relate a 1D SOT scan to a 1D
    current distribution over an infinite bar. Because, especially with noise,
    an infinite number of distributions give rise to (approximately) the same
    SOT signal, regularizers are used.

    Parameters
    ----------
    forward: Forward1D
        The forward model to use for the inversion.
    data: np.ndarray
        The SOT signal to fit, of shape (Nscan,).
    regularizer: Regularizer1D
        The regularizer to use for the inversion.
    device: torch.device
        The device to use for the inversion. Default is CPU.
    """

    def __init__(
        self,
        forward: Forward1D,
        data: np.ndarray,
        regularizer: Regularizer1D,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")

        self._forward = forward
        self._data = torch.from_numpy(data).to(device, dtype=DTYPE)
        self._regularizer = regularizer
        self._device = device

        # Initialize current with zeros
        self._current = torch.zeros(
            forward.params.Ndevice, requires_grad=True, device=self._device, dtype=DTYPE
        )

        # Optimizer pre-defined so that restarts are possible:
        self._optimizer = torch.optim.Adam([self._current], lr=1e-3)

    def _fidelity(self, signal: torch.Tensor) -> torch.Tensor:
        l2norm: torch.Tensor = torch.linalg.norm(signal - self._data) ** 2
        return l2norm

    def _objective(
        self, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Calculate (objective, fidelity, regularization, regularization_terms)."""
        signal = self._forward.forward_t(current)
        fidelity = self._fidelity(signal)
        regularizationterms = self._regularizer.terms(current)
        regularization = sum(
            regularizationterms.values(),
            start=torch.tensor(0.0, device=current.device, dtype=current.dtype),
        )
        return fidelity + regularization, fidelity, regularization, regularizationterms

    def fit(
        self,
        lr: float = 1e-3,
        n_steps: int = 20000,
        callback: FitCallback | None = None,
        callback_every: int = 500,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the current distribution to the data using gradient descent.

        Performs ADAM fit of the current distribution to the data using the
        forward model while continually regularizing. Can be restarted if
        parameter variation is required.

        Parameters
        ----------
        lr: float
            Learning rate for the ADAM optimizer. Default is 1e-3.
        n_steps: int
            Number of steps to run the optimizer. Default is 20000.
        callback: FitCallback
            Callback function to call at each (callback) step. Default is None.
        callback_every: int
            Number of steps between callback calls. Default is 500.

        Returns
        -------
        yarray_current: np.ndarray
            The y-axis of the current distribution, of shape (Ndevice,).
        j_recovered: np.ndarray
            The recovered current distribution, of shape (Ndevice,).
        loss_history: np.ndarray
            The history of the loss function, of shape (n_steps,).
        """
        # First ensure proper learning rate is set
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        optimizer = self._optimizer

        loss_history = torch.zeros(n_steps, device=self._device, dtype=DTYPE)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss, fidelity, regularization, regularization_terms = self._objective(
                self._current
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            loss_history[step] = loss.detach()

            if callback and (step % callback_every == 0 or step == n_steps - 1):
                with torch.no_grad():
                    signal = self._forward.forward_t(self._current)

                state = FitState1D(
                    step=step,
                    total_steps=n_steps,
                    loss=loss.item(),
                    fidelity=fidelity.item(),
                    regularization=regularization.item(),
                    regularization_terms={
                        k: v.item() for k, v in regularization_terms.items()
                    },
                    current=self._current.detach().cpu().numpy().copy(),
                    predicted_signal=signal.cpu().numpy().copy(),
                    to_fit_signal=self._data.detach().cpu().numpy().copy(),
                    xaxis_current=self._forward.params.ydevice.cpu().numpy().copy(),
                    xaxis_signal=self._forward.params.yscan.cpu().numpy().copy(),
                )

                if step == n_steps - 1:
                    callback.on_finish(state)
                else:
                    callback.on_step(state)

        yarray_current = self._forward.params.ydevice.cpu().numpy()
        j_recovered = self._current.detach().cpu().numpy()
        loss_history_recovered = loss_history.cpu().numpy()
        return yarray_current, j_recovered, loss_history_recovered

    @property
    def regularizer(self) -> Regularizer1D:
        """The regularizer used for the inversion."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: Regularizer1D) -> None:
        """Set the regularizer used for the inversion."""
        self._regularizer = value

    def reset(self) -> None:
        """Reset the inversion to the initial conditions."""
        self._current = torch.zeros(
            self._forward.params.Ndevice,
            requires_grad=True,
            device=self._device,
            dtype=DTYPE,
        )
        self._optimizer = torch.optim.Adam([self._current], lr=1e-3)


class Inverter1DOversampled(Inverter1D):
    """1D inverter that allows for oversampling the current distribution.

    Inverter that releases the constraint that the output of the forward
    model must be sampled on the same grid as the data. Use this class when
    the forward model has a finer grid than the data, this allows for the
    reconstruction of smaller features. To make up for the difference, every
    step, the output of the forward model gets resampled on the data grid
    to compute the fidelity term. The regularization is still computed on the
    finer grid. This adds a small overhead (on top of the forward model
    generally being more expensive due to the finer grid).

    Parameters
    ----------
    forward: Forward1D
        The forward model to use for the inversion.
    xdata: np.ndarray
        The x-axis on which the data is sampled, of shape (Nscan,).
    data: np.ndarray
        The SOT signal to fit, of shape (Nscan,).
    regularizer: Regularizer1D
        The regularizer to use for the inversion.
    device: torch.device
        The device to use for the inversion. Default is CPU.
    """

    def __init__(
        self,
        forward: Forward1D,
        xdata: np.ndarray,
        data: np.ndarray,
        regularizer: Regularizer1D,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self._xdata = torch.from_numpy(xdata).to(dtype=DTYPE, device=device)  # The
        # x-axis on which the data is sampled

        # Assert that the forward model's x-axis fully contains the data x-axis:
        x_forward = forward.params.yscan
        x_data = xdata
        assert x_forward[0] <= x_data[0] and x_forward[-1] >= x_data[-1], (
            "Forward model's x-axis must fully contain the data x-axis"
        )
        assert torch.all(self._xdata[1:] >= self._xdata[:-1]), "xdata must be sorted"

        # Precompute interpolation weights
        idx = torch.searchsorted(x_forward, self._xdata).clamp(1, len(x_forward) - 1)
        x0, x1 = x_forward[idx - 1], x_forward[idx]
        t = (self._xdata - x0) / (x1 - x0)

        self._interp_idx = idx.to(device=device)
        self._interp_t = t.to(device=device)

        super().__init__(forward, data, regularizer, device)

    def _fidelity(self, signal: torch.Tensor) -> torch.Tensor:
        signal_data = signal[self._interp_idx - 1] + self._interp_t * (
            signal[self._interp_idx] - signal[self._interp_idx - 1]
        )
        l2norm: torch.Tensor = torch.linalg.norm(signal_data - self._data) ** 2
        return l2norm

    def fit(
        self,
        lr: float = 1e-3,
        n_steps: int = 20000,
        callback: FitCallback | None = None,
        callback_every: int = 500,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits the current distribution to the data using gradient descent.

        Performs ADAM fit of the current distribution to the data using the
        forward model while continually regularizing. Can be restarted if
        parameter variation is required.

        Parameters
        ----------
        lr: float
            Learning rate for the ADAM optimizer. Default is 1e-3.
        n_steps: int
            Number of steps to run the optimizer. Default is 20000.
        callback: FitCallback
            Callback function to call at each (callback) step. Default is None.
        callback_every: int
            Number of steps between callback calls. Default is 500.

        Returns
        -------
        yarray_current: np.ndarray
            The y-axis of the current distribution, of shape (Ndevice,).
        j_recovered: np.ndarray
            The recovered current distribution, of shape (Ndevice,).
        loss_history: np.ndarray
            The history of the loss function, of shape (n_steps,).
        """
        # First ensure proper learning rate is set
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        optimizer = self._optimizer

        loss_history = torch.zeros(n_steps, device=self._device, dtype=DTYPE)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss, fidelity, regularization, regularization_terms = self._objective(
                self._current
            )
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            loss_history[step] = loss.detach()

            if callback and (step % callback_every == 0 or step == n_steps - 1):
                with torch.no_grad():
                    signal = self._forward.forward_t(self._current)
                    signal_resampled = (
                        (
                            signal[self._interp_idx - 1]
                            + self._interp_t
                            * (signal[self._interp_idx] - signal[self._interp_idx - 1])
                        )
                        .cpu()
                        .numpy()
                        .copy()
                    )

                state = FitState1D(
                    step=step,
                    total_steps=n_steps,
                    loss=loss.item(),
                    fidelity=fidelity.item(),
                    regularization=regularization.item(),
                    regularization_terms={
                        k: v.item() for k, v in regularization_terms.items()
                    },
                    current=self._current.detach().cpu().numpy().copy(),
                    predicted_signal=signal_resampled,
                    to_fit_signal=self._data.detach().cpu().numpy().copy(),
                    xaxis_current=self._forward.params.ydevice.cpu().numpy().copy(),
                    xaxis_signal=self._xdata.cpu().numpy().copy(),
                )

                if step == n_steps - 1:
                    callback.on_finish(state)
                else:
                    callback.on_step(state)

        yarray_current = self._forward.params.ydevice.cpu().numpy()
        j_recovered = self._current.detach().cpu().numpy()
        loss_history_recovered = loss_history.cpu().numpy()
        return yarray_current, j_recovered, loss_history_recovered


class LearningRateScheduler(abc.ABC):
    """Abstract base class for a learning rate scheduler to allow annealing."""

    @abc.abstractmethod
    def __call__(self, step: int) -> float:
        """Return the learning rate for the given step."""
        pass


class RegularizerScheduler(abc.ABC):
    """Abstract base class for a regularizer scheduler that enables weight annealing."""

    @abc.abstractmethod
    def __call__(self, step: int) -> Regularizer1D:
        """Return the regularizer for the given step."""
        pass


class CoinversionConvergenceChecker:
    """Live-checks convergence of the coinversion fit.

    Convergence is reached when the following four variables are constant within an
    absolute tolerance:
    1) The objective function
    2) The fidelity function
    3) The height parameter
    4) The phi_eff parameter

    The checking is performed over a window "nwindow". In all cases, the value must fall
    within tolerance over the full window, and the extrapolated value to "nextrapolate"
    steps must also fall within tolerance. This is to offer the option of filtering out
    slow drifts. Setting nextrapolate to None disables the extrapolation check. Note
    that all tolerances are absolute and dimensionful: heighttol is in the length
    unit of the forward model (micrometers for standard settings) and phitol is in
    radians. The fidelity and objective are sums (not means) of squared residuals
    in data units, so fidtol and objtol scale with the squared signal amplitude and
    with the number of scan points.

    Parameters
    ----------
    objtol : float
        Absolute tolerance for the objective function.
    fidtol : float
        Absolute tolerance for the fidelity function.
    heighttol : float
        Absolute tolerance for the height parameter.
    phitol : float
        Absolute tolerance for the phi_eff parameter.
    nwindow : int
        Number of steps to consider for convergence checking.
    nextrapolate : int | None
        Number of steps to extrapolate for convergence checking. If None, no
        extrapolation check is performed.

    Notes
    -----
    For most real data, the default absolute tolerances are too strict.
    """

    def __init__(
        self,
        objtol: float = 1e-7,
        fidtol: float = 1e-7,
        heighttol: float = 1e-4,
        phitol: float = 1e-4,
        nwindow: int = 10,
        nextrapolate: int | None = 100,
    ):
        self.objtol = objtol
        self.fidtol = fidtol
        self.heighttol = heighttol
        self.phitol = phitol
        self.nwindow = nwindow
        self.nextrapolate = nextrapolate

    def check(self, fithistory: CoinversionFitHistory) -> bool:
        """Check if convergence has been reached.

        Parameters
        ----------
        fithistory : CoinversionFitHistory
            The history of the coinversion fit.
        """
        fidelityhistory = fithistory.fidelity_history
        objectivehistory = fithistory.objective_history
        heighthistory = fithistory.height_history
        phihistory = fithistory.phi_history

        length = len(fidelityhistory)
        if length < self.nwindow:
            return False  # Not enough history to check convergence
        else:
            fidhist = np.array(fidelityhistory[-self.nwindow :])
            objhist = np.array(objectivehistory[-self.nwindow :])
            heighthist = np.array(heighthistory[-self.nwindow :])
            phihist = np.array(phihistory[-self.nwindow :])

            # Check if the values are within tolerance
            fid_converged = (np.max(fidhist) - np.min(fidhist)) < self.fidtol
            obj_converged = (np.max(objhist) - np.min(objhist)) < self.objtol
            height_converged = (
                np.max(heighthist) - np.min(heighthist)
            ) < self.heighttol
            phi_converged = (np.max(phihist) - np.min(phihist)) < self.phitol

            if not (
                fid_converged and obj_converged and height_converged and phi_converged
            ):
                return False

            # Now check the extrapolated value through a linear extrapolation for
            # another nextrapolate steps
            if self.nextrapolate is None:
                return True  # No extrapolation check requested
            x = np.arange(self.nwindow)
            fid_fit = np.polyfit(x, fidhist, 1)
            obj_fit = np.polyfit(x, objhist, 1)
            height_fit = np.polyfit(x, heighthist, 1)
            phi_fit = np.polyfit(x, phihist, 1)
            fid_extrap_value = np.polyval(fid_fit, self.nwindow + self.nextrapolate)
            obj_extrap_value = np.polyval(obj_fit, self.nwindow + self.nextrapolate)
            height_extrap_value = np.polyval(
                height_fit, self.nwindow + self.nextrapolate
            )
            phi_extrap_value = np.polyval(phi_fit, self.nwindow + self.nextrapolate)
            fid_converged = abs(fid_extrap_value - fidhist[-1]) < self.fidtol
            obj_converged = abs(obj_extrap_value - objhist[-1]) < self.objtol
            height_converged = (
                abs(height_extrap_value - heighthist[-1]) < self.heighttol
            )
            phi_converged = abs(phi_extrap_value - phihist[-1]) < self.phitol

            return bool(
                fid_converged and obj_converged and height_converged and phi_converged
            )


@dataclasses.dataclass
class CoinversionFitHistory:
    """Stores the history of the coinversion fit.

    For crossvalidation_size > 1, fidelity is the summed test-set fidelity (the CV
    score) and objective is the summed training objective of all folds; hence objective
    - fidelity is not equal to regularization.

    Parameters
    ----------
    fidelity_history : list[float]
        History of the fidelity function.
    regularization_terms_history : list[dict[str, float]]
        History of regularization, split into its individual terms.
    regularization_history : list[float]
        History of the total regularization.
    objective_history : list[float]
        History of the objective function.
    height_history : list[float]
        History of the height parameter.
    phi_history : list[float]
        History of the phi_eff parameter.
    """

    fidelity_history: list[float] = dataclasses.field(default_factory=list)
    regularization_terms_history: list[dict[str, float]] = dataclasses.field(
        default_factory=list
    )
    regularization_history: list[float] = dataclasses.field(default_factory=list)
    objective_history: list[float] = dataclasses.field(default_factory=list)
    height_history: list[float] = dataclasses.field(default_factory=list)
    phi_history: list[float] = dataclasses.field(default_factory=list)

    def append(
        self,
        fidelity: float,
        regularization_terms: dict[str, float],
        regularization: float,
        objective: float,
        height: float,
        phi: float,
    ) -> None:
        """Append a new entry to the history.

        Parameters
        ----------
        fidelity : float
            The fidelity function value.
        regularization_terms : dict[str, float]
            The individual regularization terms in a dictionary.
        regularization : float
            The total regularization value.
        objective : float
            The objective function value.
        height : float
            The height parameter value.
        phi : float
            The phi_eff parameter value.
        """
        self.fidelity_history.append(fidelity)
        self.regularization_terms_history.append(regularization_terms)
        self.regularization_history.append(regularization)
        self.objective_history.append(objective)
        self.height_history.append(height)
        self.phi_history.append(phi)


class Coinversion1D:
    """Inverter for co-fitting the sensor parameters height and phi_eff.

    Standard inversion simply minimizes the objective function, which consists of
    fidelity and regularization. Co-inversion of current and kernel parameters is much
    more complex, because there are various spurious attractors that give poor fits.
    If the kernel parameters are directly optimized alongside the current, kernels that
    allow for better-regularized currents will be preferred. For this reason, the
    problem is approached in a bilevel scheme. Moreover, in presence of noise, narrower
    kernels are preferred because they allow for overfitting. For this reason,
    a cross validation scheme can be used (by setting crossvalidation_size > 1).

    Parameters
    ----------
    forward : Forward1D
        The forward model to use for the inversion. A copy is made, such that the
        underlying parameter object can be modified. However, the parameter object
        is modified only if cache_kernels is True. The height and phi stored in
        forward.params are ignored as starting values; use initial_height and
        initial_phi_eff instead.
    data : np.ndarray
        The data to fit, of shape (Nscan,).
    regularizer: Regularizer1D | RegularizerScheduler
        The regularizer to use for the inversion. If a RegularizerScheduler is provided,
        the regularizer is updated at each sensor optimization step. Regularizers are
        deep-copied into the class and therefore disjoint from any externally kept
        reference.
    initial_height : float, optional
        The initial height parameter value. Default is 1.0.
    initial_phi_eff : float, optional
        The initial effective tilt angle in radians. Default is 0.0. This is the
        scan-frame angle Forward1DParameters.phi_eff, i.e. -phi if
        forward.params.invert_scan is True, and +phi otherwise.
    crossvalidation_size : int, optional
        The number of cross-validation splits. Default is 1. If larger than 1,
        cross-validation is used to prevent overfitting.
    convergence_checker : CoinversionConvergenceChecker | None, optional
        The convergence checker to use. Default is None.
    callback : FitCallback | None, optional
        The callback to use during fitting. Default is None.
    callback_every : int, optional
        Call callback.on_step every callback_every sensor optimization steps.
        Default is 1.
    device : torch.device | None, optional
        The device to use for computations. Default is None.
    lr_sensor : float | LearningRateScheduler, optional
        The learning rate for sensor parameters. Default is 1e-2. Learning rate
        schedulers are deep-copied and therefore disjoint from any externally kept
        reference.
    lr_current : float | LearningRateScheduler, optional
        The learning rate for current parameters. Default is 1.0. Learning rate
        schedulers are deep-copied and therefore disjoint from any externally kept
        reference.
    cache_kernels : bool, optional
        Whether to cache kernels. Default is True.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    Notes
    -----
    These parameters have the constraint that height > rho2*|sin(phi_eff)| such
    that the SQUID never crosses the sample plane. To incorporate this constraint, we
    reparametrize the height parameter as h_free and write h = exp(h_free) + rho2*
    |sin(phi_eff)|. The free parameter h_free is then optimized, and the constraint is
    automatically satisfied.

    crossvalidation_size: if >1, data is split into crossvalidation_size parts and
    training+test seperation is used to prevent overfitting. See documentation for
    further information.

    Callback behaviour: in on_step, FitState1D.current is the current of fold 0
    (fitted on the training set of fold 0 only), and predicted_signal is its
    prediction on the full scan, so held-out points are included. on_step is not
    called on the step at which convergence is detected. on_finish receives the
    final current, fitted on the full dataset, and its prediction.

    All angles, including the fitted sensor_parameters[3] and history.phi_history,
    are phi_eff in radians (see initial_phi_eff).
    """

    LBFGS_MAXREPEATS = 10
    LBFGS_TOLERANCE_GRAD = 1e-6
    LBFGS_TOLERANCE_ABS = 1e-10

    def __init__(
        self,
        forward: Forward1D,
        data: np.ndarray,
        regularizer: Regularizer1D | RegularizerScheduler,
        initial_height: float = 1.0,
        initial_phi_eff: float = 0.0,
        crossvalidation_size: int = 1,
        convergence_checker: CoinversionConvergenceChecker | None = None,
        callback: FitCallback | None = None,
        callback_every: int = 1,
        device: torch.device | None = None,
        lr_sensor: float | LearningRateScheduler = 1e-2,
        lr_current: float | LearningRateScheduler = 1.0,
        cache_kernels: bool = True,
        verbose: bool = False,
    ):
        if device is None:
            device = torch.device("cpu")

        self._forward = copy.deepcopy(forward)
        self._initial_forward = copy.deepcopy(forward)
        self._data = torch.from_numpy(data).to(device, dtype=DTYPE)
        self._regularizer = regularizer
        self._regularizer_initial = copy.deepcopy(regularizer)
        self._crossvalidation_size = crossvalidation_size
        self._convergence_checker = convergence_checker
        self._callback = callback
        self._callback_every = callback_every
        self._device = device
        self._fitted_current: np.ndarray | None = None
        self._cache_kernels = cache_kernels
        self._verbose = verbose

        self._folds = self._build_folds()

        self._lr_current = lr_current
        self._lr_current_initial = lr_current
        self._lr_sensor = lr_sensor
        self._lr_sensor_initial = lr_sensor
        self._initial_phi_eff = initial_phi_eff
        initial_relative_height = initial_height - self._forward.params.rho2 * np.abs(
            np.sin(initial_phi_eff)
        )
        if initial_relative_height <= 0:
            raise ValueError(
                f"Initial height {initial_height} is too small for the "
                f"given phi_eff {initial_phi_eff} and rho2 "
                f"{self._forward.params.rho2}. Must satisfy height > rho2 "
                f"* |sin(phi_eff)|."
            )
        self._initial_height_free = np.log(initial_relative_height)

        self.reset()

    def _build_folds(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Build the (training, test) boolean masks for every cross-validation fold.

        Fold i leaves out every K-th scan point starting at index i; those points form
        the test set, the rest the training set. Each scan point is in exactly one
        test set. K=1 is the full fit: training and test are both the entire scan.
        """
        n_points = self._data.shape[0]
        k = self._crossvalidation_size
        folds = []
        for i in range(k):
            training = torch.ones(n_points, dtype=torch.bool, device=self._device)
            if k == 1:
                test = training.clone()
            else:
                training[i::k] = False
                test = ~training
            folds.append((training, test))
        return tuple(folds)

    def _full_sensor_parameters(self, free_parameters: torch.Tensor) -> torch.Tensor:
        # Assemble the full sensor parameters tensor from the free parameters
        return torch.stack(
            [
                torch.exp(free_parameters[0])
                + self._forward.params.rho2 * torch.abs(torch.sin(free_parameters[1])),
                torch.tensor(
                    self._forward.params.rho1, device=self._device, dtype=DTYPE
                ),
                torch.tensor(
                    self._forward.params.rho2, device=self._device, dtype=DTYPE
                ),
                free_parameters[1],
            ]
        )

    def _fidelity(
        self,
        current: torch.Tensor,
        full_sensor_parameters: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cached_kernels: bool = False,
    ) -> torch.Tensor:
        signal = self._forward.forward_t(
            current, full_sensor_parameters if not use_cached_kernels else None
        )
        if mask is not None:
            signal = signal[mask]
            data = self._data[mask]
        else:
            data = self._data
        residual_norm: torch.Tensor = torch.linalg.norm(signal - data)
        return residual_norm.pow(2)

    def _objective(
        self,
        current: torch.Tensor,
        sensor_parameters_free: torch.Tensor,
        regularizer: Regularizer1D,
        mask: torch.Tensor | None = None,
        use_cached_kernels: bool = False,  # If True, cached kernels get used, however,
        # this can only work if the sensor parameters are static.
    ) -> torch.Tensor:
        full_sensor_parameters = self._full_sensor_parameters(sensor_parameters_free)
        fidelity = self._fidelity(
            current, full_sensor_parameters, mask, use_cached_kernels=use_cached_kernels
        )
        reg = regularizer(current)
        return fidelity + reg

    def _all_loss_terms(
        self,
        currents: list[torch.Tensor],
        sensor_parameters_free: torch.Tensor,
        regularizer: Regularizer1D,
    ) -> tuple[float, dict[str, float], float, float]:
        # Per fold f with current c_f:
        #   fidelity  = sum_f F_test,f(c_f)              (CV score, the outer criterion)
        #   reg       = sum_f R(c_f)
        #   objective = sum_f [F_train,f(c_f) + R(c_f)]  (summed inner objectives)
        # At K=1 (train = test = all) these reduce to F, R and F + R.
        sensor_params = self._full_sensor_parameters(sensor_parameters_free.detach())
        fidelity = 0.0
        train_fidelity = 0.0
        regterms: dict[str, float] = {}
        for current, (training_set_mask, test_set_mask) in zip(
            currents, self._folds, strict=True
        ):
            c = current.detach()
            fidelity += self._fidelity(c, sensor_params, test_set_mask).item()
            train_fidelity += self._fidelity(c, sensor_params, training_set_mask).item()
            for name, value in regularizer.terms(c).items():
                regterms[name] = regterms.get(name, 0.0) + value.item()
        reg = sum(regterms.values())
        obj = train_fidelity + reg
        return fidelity, regterms, reg, obj

    def _make_fit_state(
        self, regularizer: Regularizer1D, total_steps: int
    ) -> FitState1D:
        fidelity, regterms, reg, obj = self._all_loss_terms(
            self._currents, self._sensor_parameters_free, regularizer
        )
        return FitState1D(
            step=self._parameter_optimizer_step,
            total_steps=total_steps,
            loss=obj,
            fidelity=fidelity,
            regularization=reg,
            regularization_terms=regterms,
            current=self._fitted_current,
            predicted_signal=self.prediction,
            to_fit_signal=self.data,
            xaxis_current=self.ydevice,
            xaxis_signal=self.yscan,
            sensor_parameters=self.sensor_parameters,
        )

    def _get_regularizer(self, parameter_optimizer_step: int) -> Regularizer1D:
        return (
            self._regularizer(parameter_optimizer_step)
            if isinstance(self._regularizer, RegularizerScheduler)
            else self._regularizer
        )

    def fit(
        self,
        n_steps_sensor: int = 500,
        n_steps_current: int = 200,
    ) -> dict[str, Any]:
        """Perform or continue a full co-inversion of the dataset.

        If the fit has not been started, fits until convergence is reached, or until
        n_steps_sensor sensor optimization steps have been performed. If the fit has
        already been started (and reset() has not been called), continues where was
        left off.

        Parameters
        ----------
        n_steps_sensor : int, optional
            The number of sensor optimization steps to perform. Default is 500.
        n_steps_current : int, optional
            The number of current optimization steps to perform for each sensor
            optimization step. Default is 200.

        Returns
        -------
        dict[str, Any]
            A short fit report, containing the following keys:
            - "message" (str): A message indicating the reason for stopping the fit.
            - "steps" (int): The number of sensor optimization steps performed.
            - "steps_total" (int): The total number of sensor optimization steps
              performed since the last reset.
            - "final_sensor_parameters" (np.ndarray): The final sensor parameters after
              the fit, of shape (4,).
        """
        optimize_current_times = []
        calculate_v_times = []
        optimize_sensor_times = []
        history_times = []

        n_steps_start = self._parameter_optimizer_step

        did_converge = False

        while self._parameter_optimizer_step - n_steps_start < n_steps_sensor:
            # Update learning rate for the sensor based on the scheduler.
            lr_sensor = self._get_lr_sensor(self._parameter_optimizer_step)
            lr_current = self._get_lr_current(self._parameter_optimizer_step)
            regularizer = self._get_regularizer(self._parameter_optimizer_step)

            self._parameter_optimizer.param_groups[0]["lr"] = lr_sensor

            # Step 1: for each training set, optimize the current with sensor parameters
            # fixed.
            start_time = time.perf_counter()
            if self._cache_kernels:
                self._cache_kernels_into_forward(
                    self._sensor_parameters_free, self._forward
                )
            self._optimize_currents_fixed_sensor(
                self._currents,
                self._sensor_parameters_free,
                n_steps_current,
                lr_current,
                regularizer,
            )
            optimize_current_times.append(time.perf_counter() - start_time)
            start_time = time.perf_counter()
            vs = self._calculate_vs_at_fixed_point(
                self._currents, self._sensor_parameters_free, regularizer
            )
            calculate_v_times.append(time.perf_counter() - start_time)

            # Now we record the history:
            start_time = time.perf_counter()
            fidelity, regterms, reg, obj = self._all_loss_terms(
                self._currents, self._sensor_parameters_free, regularizer
            )
            sp = self._full_sensor_parameters(self._sensor_parameters_free.detach())
            height, phi_eff = sp[0].item(), sp[3].item()
            self._history.append(fidelity, regterms, reg, obj, height, phi_eff)

            do_callback = self._callback is not None and (
                (self._parameter_optimizer_step + 1) % self._callback_every == 0
            )
            if do_callback:
                # Sensor parameters are not yet updated in this step, so the cached
                # kernel (if any) still matches sp: one cheap FFT convolution.
                with torch.no_grad():
                    prediction_fold0 = self._forward.forward_t(
                        self._currents[0].detach(),
                        None if self._cache_kernels else sp,
                    )
                state = FitState1D(
                    step=self._parameter_optimizer_step,
                    total_steps=n_steps_start + n_steps_sensor,
                    loss=obj,
                    fidelity=fidelity,
                    regularization=reg,
                    regularization_terms=regterms,
                    current=self._currents[0].detach().cpu().numpy(),
                    predicted_signal=prediction_fold0.cpu().numpy(),
                    to_fit_signal=self.data,
                    xaxis_current=self.ydevice,
                    xaxis_signal=self.yscan,
                    sensor_parameters=sp.cpu().numpy(),
                )
            history_times.append(time.perf_counter() - start_time)

            start_time = time.perf_counter()
            self._optimize_sensor_fixed_current(
                self._currents,
                self._sensor_parameters_free,
                self._parameter_optimizer,
                vs,
                regularizer,
            )
            optimize_sensor_times.append(time.perf_counter() - start_time)

            self._parameter_optimizer_step += 1

            # Check for convergence if a checker is provided
            if self._convergence_checker is not None:
                if self._convergence_checker.check(self._history):
                    if self._verbose:
                        print(
                            f"Convergence reached at sensor optimization step "
                            f"{self._parameter_optimizer_step}."
                        )
                    did_converge = True
                    break

            # Perform live callback
            if do_callback:
                assert self._callback is not None
                self._callback.on_step(state)

        # Now that we are done, we have the proper fixed parameters to calculate the
        # current one last time. Soft start from the average current:
        if self._cache_kernels:
            self._cache_kernels_into_forward(
                self._sensor_parameters_free, self._forward
            )
        current = (
            torch.mean(torch.stack(self._currents), dim=0).detach().requires_grad_(True)
        )
        full_data_mask = torch.ones(
            self._data.shape[0], dtype=torch.bool, device=self._device
        )
        self._optimize_single_current_fixed_sensor(
            current,
            self._sensor_parameters_free,
            n_steps_current,
            self._get_lr_current(self._parameter_optimizer_step),
            self._get_regularizer(self._parameter_optimizer_step),
            full_data_mask,
        )
        self._fitted_current = np.copy(current.detach().cpu().numpy())

        # We are done, so call the finish callback if provided
        if self._callback is not None:
            shift = -1 if self._parameter_optimizer_step > 0 else 0
            state = self._make_fit_state(
                self._get_regularizer(self._parameter_optimizer_step + shift),
                total_steps=n_steps_start + n_steps_sensor,
            )
            self._callback.on_finish(state)

        # Summarize timings
        if self._verbose and optimize_current_times:
            print("FIT FINISHED --- TIMING SUMMARY")
            print(
                f"CURRENT DETERMINATION --- \tAverage: {
                    np.mean(optimize_current_times):.4f} s\t"
                f"Max: {np.max(optimize_current_times):.4f} s\t"
                f"Min: {np.min(optimize_current_times):.4f}\t"
                f"Total: {np.sum(optimize_current_times):.4f} s"
            )
            print(
                f"V CALCULATION --- \tAverage: {np.mean(calculate_v_times):.4f} s\t"
                f"Max: {np.max(calculate_v_times):.4f} s\t"
                f"Min: {np.min(calculate_v_times):.4f}\t"
                f"Total: {np.sum(calculate_v_times):.4f} s"
            )
            print(
                f"SENSOR DETERMINATION --- \tAverage: {
                    np.mean(optimize_sensor_times):.4f} s\t"
                f"Max: {np.max(optimize_sensor_times):.4f} s\t"
                f"Min: {np.min(optimize_sensor_times):.4f}\t"
                f"Total: {np.sum(optimize_sensor_times):.4f} s"
            )
            print(
                f"HISTORY RECORDING --- \t\tAverage: {np.mean(history_times):.4f} s\t"
                f"Max: {np.max(history_times):.4f} s\t"
                f"Min: {np.min(history_times):.4f}\t"
                f"Total: {np.sum(history_times):.4f} s"
            )

        # At the end, return a small dictionary reporting the final state
        final_sensor_params = self.sensor_parameters
        nsteps = self._parameter_optimizer_step - n_steps_start
        stopreason = "converged" if did_converge else "max_steps_reached"
        return {
            "message": stopreason,
            "steps": nsteps,
            "steps_total": self._parameter_optimizer_step,
            "final_sensor_parameters": final_sensor_params,
        }

    def _optimize_sensor_fixed_current(
        self,
        currents: list[torch.Tensor],
        sensor_parameters_free: torch.Tensor,
        parameter_optimizer: torch.optim.Optimizer,
        vs: list[torch.Tensor],
        regularizer: Regularizer1D,
    ) -> None:
        # Without CV, one calculates the surrogate F - Lcurr * v
        # With CV, one has as many vs as folds, and as many surrogates as folds
        # One then calculates for the ith surrogate F - Lcurr * v using:
        # - for F, the test set i
        # - for Lcurr, the training set i
        # - the ith calculated v (which comes from test set i)
        surrogates = []
        sensorparams = self._full_sensor_parameters(sensor_parameters_free)
        for i in range(self._crossvalidation_size):
            training_set_mask, test_set_mask = self._folds[i]
            curr = currents[i].clone().detach().requires_grad_(True)  # Something to
            # take derivatives with respect to.
            F_test = self._fidelity(curr, sensorparams, test_set_mask)
            F_train = self._fidelity(curr, sensorparams, training_set_mask)
            Lcurr = torch.autograd.grad(
                F_train + regularizer(curr), curr, create_graph=True
            )[0]
            surrogate = F_test - (Lcurr * vs[i]).sum()
            surrogates.append(surrogate)
        composite_surrogate: torch.Tensor = torch.stack(surrogates).sum()
        parameter_optimizer.zero_grad()
        composite_surrogate.backward(inputs=sensor_parameters_free)  # type: ignore[no-untyped-call]
        parameter_optimizer.step()

    def _calculate_vs_at_fixed_point(
        self,
        currents: list[torch.Tensor],
        sensor_parameters_free: torch.Tensor,
        regularizer: Regularizer1D,
    ) -> list[torch.Tensor]:
        # Calculates the "v" object at a fixed point in the optimization scheme. Final
        # v is detached from the graph. Each fold gets its own "v" object (we return a
        # list). Where v is calculated using only the TEST SET
        vs = []
        for i in range(self._crossvalidation_size):
            training_set_mask, test_set_mask = self._folds[i]
            v = self._calculate_single_v_at_fixed_point(
                currents[i],
                sensor_parameters_free,
                regularizer,
                training_set_mask,
                test_set_mask,
            )
            vs.append(v)
        return vs

    def _calculate_single_v_at_fixed_point(
        self,
        current: torch.Tensor,
        sensor_parameters_free: torch.Tensor,
        regularizer: Regularizer1D,
        training_set_mask: torch.Tensor,
        test_set_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Calculates the "v" object at a fixed point in the optimization scheme.
        # Final v is detached from the graph.
        # Where v is calculated using only the TEST SET
        current_fixed = current.detach()
        sensor_params_fixed = self._full_sensor_parameters(
            sensor_parameters_free
        ).detach()
        curr = current_fixed.clone().requires_grad_(True)  # Something to take
        # derivatives with respect to.
        fcurr = torch.autograd.grad(
            self._fidelity(curr, sensor_params_fixed, test_set_mask), curr
        )[0]
        hessian = torch.autograd.functional.hessian(
            lambda c: (
                self._fidelity(c, sensor_params_fixed, training_set_mask)
                + regularizer(c)
            ),
            curr,
            vectorize=True,
        )  # type: ignore[no-untyped-call]
        v: torch.Tensor = (
            torch.linalg.solve(hessian.double(), fcurr.double()).to(DTYPE).detach()
        )
        return v

    def _optimize_currents_fixed_sensor(
        self,
        currents: list[torch.Tensor],
        sensor_parameters_free: torch.Tensor,
        n_steps_current: int,
        lr_current: float,
        regularizer: Regularizer1D,
    ) -> None:
        for i in range(self._crossvalidation_size):
            training_set_mask, _ = self._folds[i]
            self._optimize_single_current_fixed_sensor(
                currents[i],
                sensor_parameters_free,
                n_steps_current,
                lr_current,
                regularizer,
                training_set_mask,
            )

    def _cache_kernels_into_forward(
        self, sensor_parameters_free: torch.Tensor, forward: Forward1D
    ) -> None:
        params = copy.deepcopy(forward.params)
        full_params = self._full_sensor_parameters(sensor_parameters_free.detach())
        h, phi_eff = full_params[0].item(), full_params[3].item()
        params.height = h
        params.phi = phi_eff if not params.invert_scan else -phi_eff
        forward.params = params  # This automatically caches, .params is a @property.

    def _optimize_single_current_fixed_sensor(
        self,
        current: torch.Tensor,
        sensor_parameters_free: torch.Tensor,
        n_steps_current: int,
        lr_current: float,
        regularizer: Regularizer1D,
        training_set_mask: torch.Tensor | None = None,
    ) -> None:
        # Current must be leaf tensor with requires_grad=True
        free_sensor_params_fixed = sensor_parameters_free.detach()
        current_optimizer = torch.optim.LBFGS(
            [current],
            lr=lr_current,
            max_iter=n_steps_current,
            max_eval=n_steps_current * 5,
            tolerance_grad=self.LBFGS_TOLERANCE_GRAD,
            tolerance_change=self.LBFGS_TOLERANCE_ABS,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            current_optimizer.zero_grad()
            loss = self._objective(
                current,
                free_sensor_params_fixed,
                regularizer,
                training_set_mask,
                use_cached_kernels=self._cache_kernels,
            )
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        # As long as max_steps is reached, continue, to ensure we end up converged
        st = current_optimizer.state[current]
        n_before = 0  # fresh optimizer
        current_optimizer.step(closure)  # type: ignore[no-untyped-call]
        nrepeats = 1
        while st["n_iter"] - n_before == n_steps_current:
            n_before = st["n_iter"]
            current_optimizer.step(closure)  # type: ignore[no-untyped-call]
            nrepeats += 1
            if nrepeats > self.LBFGS_MAXREPEATS:
                warnings.warn(
                    "LBFGS did not converge within the maximum number of repeats. "
                    "Consider increasing n_steps_current or adjusting lr_current. "
                    "Alternatively, perhaps the tolerance is set too strictly.",
                    stacklevel=2,
                )
                break

    def _get_lr_sensor(self, parameter_optimizer_step: int) -> float:
        lr_sensor = (
            self._lr_sensor(parameter_optimizer_step)
            if isinstance(self._lr_sensor, LearningRateScheduler)
            else self._lr_sensor
        )
        return lr_sensor

    def _get_lr_current(self, parameter_optimizer_step: int) -> float:
        lr_current = (
            self._lr_current(parameter_optimizer_step)
            if isinstance(self._lr_current, LearningRateScheduler)
            else self._lr_current
        )
        return lr_current

    def reset(self) -> None:
        """Reset the state of the fit to return to the initial state.

        Performs a true "factory reset" as if fit() was never run.
        """
        self._parameter_optimizer_step = 0
        self._forward = copy.deepcopy(self._initial_forward)

        self._lr_current = self._lr_current_initial
        self._regularizer = copy.deepcopy(self._regularizer_initial)

        self._currents = [
            torch.zeros(
                self._forward.params.Ndevice,
                requires_grad=True,
                device=self._device,
                dtype=DTYPE,
            )
            for _ in range(self._crossvalidation_size)
        ]  # List of currents, one predicted
        # by each training set. The final prediction should be made by using the final
        # kernel and fitting on a training set spanning the entire data set.
        self._fitted_current = None

        # Initialize the free sensor parameters (height and phi_eff) with the provided
        # initial values
        self._sensor_parameters_free = torch.tensor(
            [self._initial_height_free, self._initial_phi_eff],
            requires_grad=True,
            device=self._device,
            dtype=DTYPE,
        )
        self._lr_sensor = self._lr_sensor_initial
        lr_sensor = self._get_lr_sensor(
            self._parameter_optimizer_step
        )  # Should be 0 :)
        self._parameter_optimizer = torch.optim.Adam(
            [self._sensor_parameters_free], lr=lr_sensor
        )

        self._history = CoinversionFitHistory()

    @property
    def lr_sensor(self) -> float | LearningRateScheduler:
        """The learning rate for the sensor parameters, potentially a scheduler."""
        return self._lr_sensor

    @lr_sensor.setter
    def lr_sensor(self, value: float | LearningRateScheduler) -> None:
        # Setting the learning rate to a float when a scheduler is provided
        # should raise. However, changing the scheduler mid-run is allowed.
        if isinstance(self._lr_sensor, LearningRateScheduler) and (
            not isinstance(value, LearningRateScheduler)
        ):
            raise ValueError(
                "Cannot set lr_sensor to a float when a "
                "LearningRateScheduler is in use. Use a scheduler or set "
                "the initial learning rate to a float."
            )
        self._lr_sensor = value

    @property
    def lr_current(self) -> float | LearningRateScheduler:
        """The learning rate for the current optimization, potentially a scheduler."""
        return self._lr_current

    @lr_current.setter
    def lr_current(self, value: float | LearningRateScheduler) -> None:
        if isinstance(self._lr_current, LearningRateScheduler) and (
            not isinstance(value, LearningRateScheduler)
        ):
            raise ValueError(
                "Cannot set lr_current to a float when a "
                "LearningRateScheduler is in use. Use a scheduler or set "
                "the initial learning rate to a float."
            )
        self._lr_current = value

    @property
    def regularizer(self) -> Regularizer1D | RegularizerScheduler:
        """The regularizer used for regularization, either a function or scheduler."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, value: Regularizer1D | RegularizerScheduler) -> None:
        # Setting the regularizer to a Regularizer1D when a RegularizerScheduler is
        # provided should raise. However, changing the scheduler mid-run is allowed.
        if isinstance(self._regularizer, RegularizerScheduler) and (
            not isinstance(value, RegularizerScheduler)
        ):
            raise ValueError(
                "Cannot set regularizer to a Regularizer1D when a "
                "RegularizerScheduler is in use. Use a scheduler or set "
                "the initial regularizer to a Regularizer1D."
            )
        self._regularizer = value

    @property
    def history(self) -> CoinversionFitHistory:
        """The history of the fit."""
        return self._history

    @property
    def sensor_parameters(self) -> np.ndarray:
        """The fitted sensor parameters (height, rho1, rho2, phi_eff)."""
        sps = self._full_sensor_parameters(self._sensor_parameters_free)
        return np.copy(sps.detach().cpu().numpy())

    @property
    def ydevice(self) -> np.ndarray:
        """Axis over which the current profile is defined (the device axis)."""
        return np.copy(self._forward.params.ydevice.detach().cpu().numpy())

    @property
    def current(self) -> np.ndarray | None:
        """The fitted current profile calculated from the full dataset."""
        return self._fitted_current

    @property
    def yscan(self) -> np.ndarray:
        """Axis over which the (data) scan was performed."""
        return np.copy(self._forward.params.yscan.detach().cpu().numpy())

    @property
    def data(self) -> np.ndarray:
        """The raw data meant to fit."""
        return np.copy(self._data.detach().cpu().numpy())

    @property
    def prediction(self) -> np.ndarray:
        """The predicted signal corresponding to the last-fit current."""
        if self.current is None:
            raise ValueError("No current has been fitted yet. Call fit() first.")
        with torch.no_grad():
            sensorparams = self._full_sensor_parameters(self._sensor_parameters_free)
            return np.copy(
                self._forward.forward_t(
                    torch.from_numpy(self.current).to(self._device, dtype=DTYPE),
                    sensorparams,
                )
                .detach()
                .cpu()
                .numpy()
            )

    @property
    def convergence_checker(self) -> CoinversionConvergenceChecker | None:
        """The object that checks for convergence of the fit."""
        return self._convergence_checker

    @convergence_checker.setter
    def convergence_checker(self, value: CoinversionConvergenceChecker | None) -> None:
        self._convergence_checker = value


class Inverter2D:
    """Invert a 2D SOT signal to the underlying current distribution.

    Inverter for full 2D images. Unlike the 1D inverter, 2D inversion
    uses a mesh to represent the sample. This enables accurate modeling of
    sample geometry, which can be extracted from AFM, SEM, litho pattern, etc.

    Parameters
    ----------
    meshtogrid: MeshToGrid
        The mesh to grid converter that encapsulates the mesh and defines the
        operation that converts the mesh to a grid for the forward model. It
        is the user's responsibility to ensure the output of meshtogrid is
        compatible with the forward model.
    currentforward: Forward2DCurrent
        The forward model that computes the SOT signal from the current
        distribution on the mesh. It is the user's responsibility to ensure
        the output of meshtogrid is compatible with the forward model.
    data: np.ndarray
        The SOT signal to fit, of shape (Nscan_y, Nscan_x).
    lam1: float
        Regularization weight for Tikhonov regularization on the
        streamfunction.
    lam2: float
        Regularization weight for total variation regularization on the
        current density.
    lam3: float
        Regularization weight for 3rd order Tikhonov regularization on the
        streamfunction, equivalent to 1st order Tikhonov on the vorticity.
    huber_e: float
        Huber parameter for total variation regularization. Default is 1e-3.
    lumped_mass: bool, optional
        Whether to use a lumped mass matrix for the Tikhonov regularization.
        Default is False, however, on fine meshes, the lumped mass matrix is
        many times faster, and lumped_mass=False can stall the inversion.
    initial_streamfunction: np.ndarray, optional
        Initial guess for the streamfunction, of shape (Nmesh_vertices,). If not
        supplied, the initial guess is zero everywhere.
    device: torch.device, optional
        The device to use for the inversion. Default is CPU.
    """

    LBFGS_MAXREPEATS = 10
    LBFGS_TOLERANCE_GRAD = 1e-6
    LBFGS_TOLERANCE_ABS = 1e-10

    def __init__(
        self,
        meshtogrid: MeshToGrid,
        currentforward: Forward2DCurrent,
        data: np.ndarray,
        lam1: float = 1e-7,  # Tikhonov-2 on streamfunction
        lam2: float = 1e-9,  # TV on J
        lam3: float = 1e-12, # Tikhonov-3 on streamfunction
        huber_e: float = 1e-3,  # Huber parameter for TV
        lumped_mass: bool = False, # TODO: add lumped_mass=True to the test suite.
        initial_streamfunction: np.ndarray | None = None,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self._data = torch.from_numpy(data).to(dtype=DTYPE, device=device)

        self._meshtogrid = meshtogrid
        self._mesh = meshtogrid.mesh
        self._currentforward = currentforward
        self._lam1 = lam1
        self._lam2 = lam2
        self._lam3 = lam3
        self._huber_e = huber_e
        self._device = device
        self._lumped_mass = lumped_mass

        # Mesh properties
        self._N_points = len(self._mesh.vertices)
        self._N_free_vals = np.sum(self._mesh.free_mask)
        self._N_solution = (
            1 + self._N_free_vals
        )  # 1 for the Dirichlet value on the right boundary
        self._free_mask = torch.from_numpy(self._mesh.free_mask).to(device)
        self._free_idx = torch.nonzero(self._free_mask, as_tuple=True)[0]
        self._dirichlet_left_mask = torch.from_numpy(self._mesh.dirichletleft_mask).to(
            device
        )
        self._dirichlet_right_mask = torch.from_numpy(
            self._mesh.dirichletright_mask
        ).to(device)
        self._edge_lengths = torch.from_numpy(self._mesh.internaledgelengths).to(
            device, dtype=DTYPE
        )

        # Tikhonov matrices
        # This matrix is K^F,T M^FF,-1 K^F
        self._stiffness_matrix = self._mesh.stiffness_matrix
        self._mass_matrix = scipy.sparse.csr_matrix(self._mesh.mass_matrix)
        K_rows_free = self._stiffness_matrix[self._mesh.free_idx, :]  # (N_free, N_all)
        if not lumped_mass:
            Mff = scipy.sparse.csr_matrix(
                self._mass_matrix[np.ix_(self._mesh.free_idx, self._mesh.free_idx)]
            )  # (N_free, N_free)
            Kt_Minv_K = (
                K_rows_free.tocsc()
                .transpose()
                .dot(scipy.sparse.linalg.spsolve(Mff, K_rows_free.tocsc()))
            )  # (N_all, N_all)
            self._tikhonov_matrix_dense = torch.from_numpy(Kt_Minv_K.toarray()).to(
                device, dtype=DTYPE
            )
            self._tikhonov_matrix_sparse = None
        else:
            # Lump by make each diagnonal element the row sum
            Mff = scipy.sparse.csr_matrix(
                self._mass_matrix[np.ix_(self._mesh.free_idx, self._mesh.free_idx)]
            )  # (N_free, N_free)
            Mff_diag = np.array(Mff.sum(axis=1)).flatten()  # (N_free,)
            Kt_Minv_K = K_rows_free.T @ (
                scipy.sparse.diags(1.0 / Mff_diag) @ K_rows_free
            )
            self._tikhonov_matrix_sparse = self._scipy_csr_to_torch_sparse(Kt_Minv_K)
            self._tikhonov_matrix_dense = None

        # TV matrices
        DGx = (
            self._mesh.edgedifference_operator @ self._mesh.gradient_operator_x
        ).tocoo()
        self._DGx = self._scipy_csr_to_torch_sparse(DGx)
        DGy = (
            self._mesh.edgedifference_operator @ self._mesh.gradient_operator_y
        ).tocoo()
        self._DGy = self._scipy_csr_to_torch_sparse(DGy)

        # Vortex regularization matrices
        # This matrix has the shape (K^F)^T M^FF,-1 K^FF M^FF,-1 K^F
        # M^FF,-1 can be used exact or lumped.
        Kf = K_rows_free
        Kff = self._stiffness_matrix[np.ix_(self._mesh.free_idx, self._mesh.free_idx)]
        # Mff is already in memory from the Tikhonov calculation above
        if not lumped_mass:
            Kt_Minv_Kff_Minv_K = Kf.T @ (
                scipy.sparse.linalg.spsolve(
                    Mff, Kff @ scipy.sparse.linalg.spsolve(Mff, Kf)
                )
            )
            self._vorticity_matrix_dense = torch.from_numpy(
                Kt_Minv_Kff_Minv_K.toarray()
            ).to(device, dtype=DTYPE)
            self._vorticity_matrix_sparse = None
        else:
            Mff_diag = np.array(Mff.sum(axis=1)).flatten()  # (N_free,)
            Kt_Minv_Kff_Minv_K = Kf.T @ (scipy.sparse.diags(1.0 / Mff_diag) @ Kff @ scipy.sparse.diags(1.0 / Mff_diag) @ Kf)
            self._vorticity_matrix_sparse = self._scipy_csr_to_torch_sparse(Kt_Minv_Kff_Minv_K)
            self._vorticity_matrix_dense = None

        # Initial solution vector: [right-wall value, free-node values].
        # The user supplies a full streamfunction (N_vertices,); convert it to this layout.
        N_solution = 1 + int(np.sum(self._mesh.free_mask))
        solution0 = torch.zeros(N_solution, dtype=DTYPE, device=device)
        if initial_streamfunction is not None:
            g0 = np.asarray(initial_streamfunction, dtype=float)
            if g0.shape != (self._N_points,):
                raise ValueError(
                    f"initial_streamfunction must have shape ({self._N_points},), "
                    f"got {g0.shape}."
                )
            if not np.allclose(g0[self._mesh.dirichletleft_mask], 0.0):
                raise ValueError(
                    "initial_streamfunction must be zero on the left Dirichlet boundary."
                )
            right_values = g0[self._mesh.dirichletright_mask]
            if not np.allclose(right_values, right_values[0]):
                raise ValueError(
                    "initial_streamfunction must be constant on the right Dirichlet boundary."
                )
            solution0[0] = float(right_values[0])
            solution0[1:] = torch.from_numpy(g0[self._mesh.free_mask]).to(
                dtype=DTYPE, device=device
            )
        self._solution = torch.nn.Parameter(solution0)

    def _scipy_csr_to_torch_sparse(self, csr: scipy.sparse.csr_matrix) -> torch.Tensor:
        coo = csr.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        values = torch.from_numpy(coo.data).to(dtype=DTYPE)
        return torch.sparse_coo_tensor(
            indices, values, size=coo.shape, device=self._device
        ).coalesce()

    def _solution_to_streamvector(self) -> torch.Tensor:
        N_points = len(self._mesh.vertices)

        streamvector = torch.zeros(
            N_points, dtype=self._solution.dtype, device=self._device
        )
        streamvector[self._free_mask] = self._solution[1:]
        streamvector[self._dirichlet_left_mask] = 0.0
        streamvector[self._dirichlet_right_mask] = self._solution[0]
        return streamvector

    def _fidelity(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            prediction, self._data.to(dtype=prediction.dtype)
        )

    def _tikhonov_regularization(self, streamvector: torch.Tensor) -> torch.Tensor:
        """Tikhonov regularization on the streamfunction.

        For a mass matrix M and a stiffness matrix K, the Tikhonov regularizer
        is given by: R = streamvector^T K^T M^-1 K streamvector. We should
        only weigh interior nodes, however, so that the total current can
        freely change.
        """
        if self._lumped_mass:
            return torch.dot(streamvector, self._tikhonov_matrix_sparse @ streamvector)
        else:
            return torch.dot(streamvector, self._tikhonov_matrix_dense @ streamvector)

    def _tv_regularization(self, streamvector: torch.Tensor) -> torch.Tensor:
        """Total variation regularization on the current density J.

        The TV regularizer is given by: R = sum_e l_e ||J_e||
        Here J_e is the jump in the current density over edge e with length l_e
        """
        jump_x = self._DGx @ streamvector
        jump_y = self._DGy @ streamvector
        jump_mag_sq = jump_x**2 + jump_y**2
        huber_mag = torch.where(
            jump_mag_sq < self._huber_e**2,
            0.5 * jump_mag_sq / self._huber_e,
            torch.sqrt(jump_mag_sq.clamp(min=self._huber_e**2)) - 0.5 * self._huber_e,
        )

        return torch.dot(self._edge_lengths, huber_mag)

    def _vorticity_regularization(self, streamvector: torch.Tensor) -> torch.Tensor:
        if self._vorticity_matrix_sparse is not None:
            return torch.dot(streamvector, self._vorticity_matrix_sparse @ streamvector)
        else:
            return torch.dot(streamvector, self._vorticity_matrix_dense @ streamvector)

    def _objective(
        self, solution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        streamvector = self._solution_to_streamvector()
        signal = self._currentforward.forward_t(
            *self._meshtogrid.streamfunction_to_currents_t(streamvector)
        )

        fidelity = self._fidelity(signal)
        tikhonov_reg = self._tikhonov_regularization(streamvector)
        tv_reg = self._tv_regularization(streamvector)
        vor_reg = self._vorticity_regularization(streamvector)

        return (
            fidelity +
            self._lam1 * tikhonov_reg +
            self._lam2 * tv_reg +
            self._lam3 * vor_reg,
            fidelity,
            tikhonov_reg,
            tv_reg,
            vor_reg
        )

    def fit(
        self,
        lr: float = 1,
        n_steps: int = 200,
    ) -> dict[str, typing.Any]:
        """Fit the current distribution to the data using gradient descent.

        Performs L-BFGS fit of the current distribution to the data using the
        forward model while continually regularizing. Can be restarted if
        parameter variation is required.

        Parameters
        ----------
        lr: float
            Learning rate for the L-BFGS optimizer. Default is 1.
        n_steps: int
            Number of steps to run the optimizer. Default is 200.

        Returns
        -------
        res: dict
            Dictionary containing the following keys:
            - "streamfunction": np.ndarray of shape (N_points,) representing
              the recovered streamfunction on the mesh.
            - "Kx": np.ndarray of shape (N_triangles,) representing the
              recovered current density in the x-direction on the cell centers.
            - "Ky": np.ndarray of shape (N_triangles,) representing the
              recovered current density in the y-direction on the cell centers.
            - "mesh": Mesh object representing the mesh used for the inversion.
            - "loss_history": np.ndarray  representing the history of the loss function
               during the fit.
            - "fidelity_history": np.ndarray of shape (n_steps,) representing the
               history of the fidelity term during the fit.
        """
        loss_history: list[float] = []
        fidelity_history: list[float] = []

        current_optimizer = torch.optim.LBFGS(
            [self._solution],
            lr=lr,
            max_iter=n_steps,
            max_eval=n_steps * 5,
            tolerance_grad=self.LBFGS_TOLERANCE_GRAD,
            tolerance_change=self.LBFGS_TOLERANCE_ABS,
            line_search_fn="strong_wolfe"
        )

        norm_constant = torch.mean(self._data**2).detach()  # For normalizing the loss history
        def closure() -> torch.Tensor:
            current_optimizer.zero_grad()
            loss, fidelity, *_ = self._objective(self._solution)
            loss_history.append(loss.item())
            fidelity_history.append(fidelity.item())
            normalized_loss = loss / norm_constant
            normalized_loss.backward()  # type: ignore[no-untyped-call]
            return normalized_loss

        # As long as max_steps is reached, continue, to ensure we end up converged
        st = current_optimizer.state[self._solution]
        n_before = 0  # fresh optimizer
        current_optimizer.step(closure)  # type: ignore[no-untyped-call]
        nrepeats = 1
        while st["n_iter"] - n_before == n_steps:
            n_before = st["n_iter"]
            current_optimizer.step(closure)  # type: ignore[no-untyped-call]
            nrepeats += 1
            if nrepeats > self.LBFGS_MAXREPEATS:
                warnings.warn(
                    "LBFGS did not converge within the maximum number of repeats. "
                    "Consider increasing n_steps or adjusting lr. "
                    "Alternatively, perhaps the tolerance is set too strictly.",
                    stacklevel=2,
                )
                break

        streamfunction = self._solution_to_streamvector().detach().cpu().numpy()
        Kx, Ky = self._mesh.field_curl(streamfunction)

        res = {
            "streamfunction": streamfunction,
            "Kx": Kx,
            "Ky": Ky,
            "mesh": self._mesh,
            "loss_history": np.array(loss_history),
            "fidelity_history": np.array(fidelity_history),
        }

        return res

    @property
    def lam1(self) -> float:
        """Weight of Tikhonov regularization."""
        return self._lam1

    @lam1.setter
    def lam1(self, value: float) -> None:
        """Set the weight of Tikhonov regularization."""
        self._lam1 = value

    @property
    def lam2(self) -> float:
        """Weight of total variation regularization."""
        return self._lam2

    @lam2.setter
    def lam2(self, value: float) -> None:
        """Set the weight of total variation regularization."""
        self._lam2 = value

    @property
    def lam3(self) -> float:
        """Weight of vorticity regularization."""
        return self._lam3

    @lam3.setter
    def lam3(self, value: float) -> None:
        """Set the weight of vorticity regularization."""
        self._lam3 = value
