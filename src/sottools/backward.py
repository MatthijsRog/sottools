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

import dataclasses
import sys
import typing
from typing import Protocol

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import torch

from sottools.forward import Forward1D, Forward2DCurrent, MeshToGrid

DTYPE = torch.float32
type _Ax = matplotlib.axes.Axes
type _Fig = matplotlib.figure.Figure


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
    """

    step: int
    total_steps: int
    loss: float
    fidelity: float
    regularization: float
    current: np.ndarray | None
    predicted_signal: np.ndarray | None
    to_fit_signal: np.ndarray | None
    xaxis_current: np.ndarray | None
    xaxis_signal: np.ndarray | None


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
        IPython.display.clear_output(wait=True)  # type: ignore[no-untyped-call]

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
        IPython.display.display(self._fig)  # type: ignore[no-untyped-call]

    def on_finish(self, state: FitState1D) -> None:
        """Call after the last step of a 1D fit.

        Parameters
        ----------
        state : FitState1D
            State of the 1D fitting process at this step.
        """
        self.on_step(state)
        plt.close(self._fig)


class Inverter1D:
    """Inverts a 1D SOT signal to find the corresponding current distribution.

    Uses the Forward1D model in forward.py to relate a 1D SOT scan to a 1D
    current distribution over an infinite bar. Because, especially with noise,
    an infinite number of distributions give rise to (approximately) the same
    SOT signal, regularizers are used. The current implementation of the class
    uses TV-1, TV-2, and positivity regularizers. Perhaps, in the future,
    Tikhonov can be added.

    Parameters
    ----------
    forward: Forward1D
        The forward model to use for the inversion.
    data: np.ndarray
        The SOT signal to fit, of shape (Nscan,).
    lam1: float
        Regularization weight for 1st order TV regularization. Default is 0.0.
    lam2: float
        Regularization weight for 2nd order TV regularization. Default is 0.0.
    lam3: float
        Regularization weight for positivity regularization. Default is 0.0.
    lam4: float
        Regularization weight for edge ringing suppression (through further
        requirement of positivity). Default is 0.0.
    n_edge: int
         Number of points on each edge to apply the edge ringing suppression.
    device: torch.device
        The device to use for the inversion. Default is CPU.
    """

    def __init__(
        self,
        forward: Forward1D,
        data: np.ndarray,
        lam1: float = 0.0,
        lam2: float = 0.0,
        lam3: float = 0.0,
        lam4: float = 0.0,
        n_edge: int = 3,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")

        self._forward = forward
        self._data = torch.from_numpy(data).to(device, dtype=DTYPE)
        self._lam1 = lam1  # 1st order TV
        self._lam2 = lam2  # 2nd order TV
        self._lam3 = lam3  # Positivity
        self._lam4 = lam4  # Edge ringing suppression
        self._n_edge = n_edge  # Edge ringing suppression width
        self._device = device

        self._data = self._data.to(self._device)

        # Initialize current with zeros
        self._current = torch.zeros(
            forward.params.Ndevice, requires_grad=True, device=self._device
        )

        # Optimizer pre-defined so that restarts are possible:
        self._optimizer = torch.optim.Adam([self._current], lr=1e-3)

    def _fidelity(self, signal: torch.Tensor) -> torch.Tensor:
        l2norm: torch.Tensor = torch.linalg.norm(signal - self._data) ** 2
        return l2norm

    def _regularization(self, current: torch.Tensor) -> torch.Tensor:
        regularizer: torch.Tensor = torch.tensor(0.0, device=current.device)

        # 1st order TV:
        if self._lam1 > 0.0:
            tv = torch.sum(torch.abs(current[1:] - current[:-1]))
            regularizer += self._lam1 * tv

        # 2nd order TV:
        if self._lam2 > 0.0:
            tv = torch.sum(torch.abs(current[2:] - 2 * current[1:-1] + current[:-2]))
            regularizer += self._lam2 * tv

        # Positivity:
        if self._lam3 > 0.0:
            positivity_penalty = torch.sum(torch.relu(-current) ** 2)
            regularizer += self._lam3 * positivity_penalty

        if self._lam4 > 0.0:
            n = self._n_edge
            edges = torch.cat([current[:n], current[-n:]])
            edge_penalty = torch.sum(torch.nn.functional.relu(-edges) ** 2)
            regularizer += self._lam4 * edge_penalty

        return regularizer

    def _objective(
        self, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate (objective, fidelity, regularization)."""
        signal = self._forward.forward_t(current)
        fidelity = self._fidelity(signal)
        regularization = self._regularization(current)
        return fidelity + regularization, fidelity, regularization

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

        loss_history = torch.zeros(n_steps, device=self._device)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss, fidelity, regularization = self._objective(self._current)
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
    def lam1(self) -> float:
        """Weight of 1st order TV regularization."""
        return self._lam1

    @lam1.setter
    def lam1(self, value: float) -> None:
        """Set the weight of 1st order TV regularization."""
        self._lam1 = value

    @property
    def lam2(self) -> float:
        """Weight of 2nd order TV regularization."""
        return self._lam2

    @lam2.setter
    def lam2(self, value: float) -> None:
        """Set the weight of 2nd order TV regularization."""
        self._lam2 = value

    @property
    def lam3(self) -> float:
        """Weight of positivity regularization."""
        return self._lam3

    @lam3.setter
    def lam3(self, value: float) -> None:
        """Set the weight of positivity regularization."""
        self._lam3 = value

    @property
    def lam4(self) -> float:
        """Weight of edge ringing suppression regularization."""
        return self._lam4

    @lam4.setter
    def lam4(self, value: float) -> None:
        """Set the weight of edge ringing suppression regularization."""
        self._lam4 = value

    @property
    def n_edge(self) -> int:
        """Number of points on each edge to apply the edge ringing suppression."""
        return self._n_edge

    @n_edge.setter
    def n_edge(self, value: int) -> None:
        """Set the number of edge points to apply ringing suppression."""
        self._n_edge = value

    def reset(self) -> None:
        """Reset the inversion to the initial conditions."""
        self._current = torch.zeros(
            self._forward.params.Ndevice, requires_grad=True, device=self._device
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
    lam1: float
        Regularization weight for 1st order TV regularization. Default is 0.0.
    lam2: float
        Regularization weight for 2nd order TV regularization. Default is 0.0.
    lam3: float
        Regularization weight for positivity regularization. Default is 0.0.
    lam4: float
        Regularization weight for edge ringing suppression (through further
        requirement of positivity). Default is 0.0.
    n_edge: int
        Number of points on each edge to apply the edge ringing suppression.
    device: torch.device
        The device to use for the inversion. Default is CPU.
    """

    def __init__(
        self,
        forward: Forward1D,
        xdata: np.ndarray,
        data: np.ndarray,
        lam1: float = 0.0,
        lam2: float = 0.0,
        lam3: float = 0.0,
        lam4: float = 0.0,
        n_edge: int = 3,
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

        super().__init__(forward, data, lam1, lam2, lam3, lam4, n_edge, device)

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

        loss_history = torch.zeros(n_steps, device=self._device)

        for step in range(n_steps):
            optimizer.zero_grad()
            loss, fidelity, regularization = self._objective(self._current)
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
    huber_e: float
        Huber parameter for total variation regularization. Default is 1e-3.
    device: torch.device
        The device to use for the inversion. Default is CPU.
    """

    def __init__(
        self,
        meshtogrid: MeshToGrid,
        currentforward: Forward2DCurrent,
        data: np.ndarray,
        lam1: float = 1e-7,  # Tikhonov-1 on streamfunction
        lam2: float = 1e-9,  # TV on J
        huber_e: float = 1e-3,  # Huber parameter for TV
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
        self._huber_e = huber_e
        self._device = device

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
        self._stiffness_matrix = self._mesh.stiffness_matrix
        self._mass_matrix = scipy.sparse.csr_matrix(self._mesh.mass_matrix)
        K_rows_free = self._stiffness_matrix[self._mesh.free_idx, :]  # (N_free, N_all)
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

        # TV matrices
        self._DGx = torch.tensor(
            (
                self._mesh.edgedifference_operator @ self._mesh.gradient_operator_x
            ).toarray(),
            dtype=DTYPE,
            device=device,
        )
        self._DGy = torch.tensor(
            (
                self._mesh.edgedifference_operator @ self._mesh.gradient_operator_y
            ).toarray(),
            dtype=DTYPE,
            device=device,
        )

        # Initialize streamfunction with zeros
        N_free_vals = np.sum(self._mesh.free_mask)
        N_solution = 1 + N_free_vals  # 1 for the Dirichlet value on the right boundary
        self._solution = torch.nn.Parameter(
            torch.zeros(N_solution, dtype=DTYPE, device=device)
        )

        # Optimizer pre-defined so that restarts are possible:
        self._optimizer = torch.optim.Adam([self._solution], lr=1e-2)

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

    def _objective(
        self, solution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        streamvector = self._solution_to_streamvector()
        signal = self._currentforward.forward_t(
            *self._meshtogrid.streamfunction_to_currents_t(streamvector)
        )

        fidelity = self._fidelity(signal)
        tikhonov_reg = self._tikhonov_regularization(streamvector)
        tv_reg = self._tv_regularization(streamvector)

        return (
            fidelity + self._lam1 * tikhonov_reg + self._lam2 * tv_reg,
            fidelity,
            tikhonov_reg,
            tv_reg,
        )

    def fit(
        self,
        lr: float = 1e-2,
        n_steps: int = 20000,
        callback: FitCallback | None = None,
        callback_every: int = 500,
    ) -> dict[str, typing.Any]:
        """Fit the current distribution to the data using gradient descent.

        Performs ADAM fit of the current distribution to the data using the
        forward model while continually regularizing. Can be restarted if
        parameter variation is required.

        Parameters
        ----------
        lr: float
            Learning rate for the ADAM optimizer. Default is 1e-2.
        n_steps: int
            Number of steps to run the optimizer. Default is 20000.
        callback: FitCallback
            Callback function to call at each (callback) step. Default is None.
        callback_every: int
            Number of steps between callback calls. Default is 500.

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
            - "loss_history": np.ndarray of shape (n_steps,) representing the
              history of the loss function during the fit.
            - "fidelity_history": np.ndarray of shape (n_steps,) representing
              the history of the fidelity term during the fit.
        """
        # First ensure proper learning rate is set
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        optimizer = self._optimizer

        loss_history = torch.zeros(n_steps, device=self._device)
        fidelity_history = torch.zeros(n_steps, device=self._device)

        for step in range(n_steps):
            optimizer.zero_grad()

            loss, fidelity, tikhonov_reg, tv_reg = self._objective(self._solution)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            loss_history[step] = loss.detach()
            fidelity_history[step] = fidelity.detach()

            if callback and (step % callback_every == 0 or step == n_steps - 1):
                state = FitState1D(
                    step=step,
                    total_steps=n_steps,
                    loss=loss.item(),
                    fidelity=fidelity.item(),
                    regularization=(
                        self._lam1 * tikhonov_reg + self._lam2 * tv_reg
                    ).item(),
                    current=None,
                    predicted_signal=None,
                    to_fit_signal=None,
                    xaxis_current=None,
                    xaxis_signal=None,
                )

                if step == n_steps - 1:
                    callback.on_finish(state)
                else:
                    callback.on_step(state)

        streamfunction = self._solution_to_streamvector().detach().cpu().numpy()
        Kx, Ky = self._mesh.field_curl(streamfunction)

        res = {
            "streamfunction": streamfunction,
            "Kx": Kx,
            "Ky": Ky,
            "mesh": self._mesh,
            "loss_history": loss_history.detach().cpu().numpy(),
            "fidelity_history": fidelity_history.detach().cpu().numpy(),
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
