"""Solvers for forward problems in SQUID-based current imaging.

This module contains various classes to parametrize, accelerate and solve the
calculation of SQUID microscopy signals from 1D and 2D current distributions.

For kernel plotting functions, we refer to forwardplotting.py.
"""

import math
from collections.abc import Iterable

import matplotlib.tri as mtri
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm

from sottools.mesh import SimplyConnectedCurrentMesh

DTYPE = torch.float32


class Forward1DParameters:
    """Parameters describing a 1D forwrad calculation from current to SOT image.

    The forward pass of the 1D model relies on parameters describing:
    - The spatial resolution and constraints of the model
    - The properties of the SQUID sensor modeled
    - The properties of the 1D underlying current distribution

    Not every parameter can be set manually, and sometimes the model adjusts
    parameters upon setting. This is to ensure that the forward
    model always abides by these rules:

    1. All spatial axes are symmetric around zero.
    2. Number of points N is even.
    3. The y-axis runs from -L/2 to L/2-dy, where L is the length of the y-axis
    and dy is the step.

    The simplest way to ensure these rules are abided by is to make the
    following constraints and assumptions:

    1. The user sets L and N, dy is calculated as L/N and odd N is rejected.
    2. The user can choose to specify the current distribution on a model-
       defined y-axis, or provide both an y-axis and distrbution which will
       be resampled.
    3. Gamma is not pi/2 or -pi/2

    The forward model takes into account the angle gamma between the SQUID
    scan directiona and the current direction. This means that there is no
    direct convolution between the current distribution and the SQUID response
    curve; instead, the convolution converts the current distribution
    into a SQUID curve along a projected y-axis y'' related to the scanned
    y-axis by y'' = y * cos(gamma). To calculate SQUID signals, the projected
    signal is first calculated, upon which the x-axis is rescaled.

    Parameters
    ----------
    Lscan: float
        Length of the scanned y-axis on which the SQUID signal is calculated.
    Wdevice: float
        Length of the device y-axis on which the current distribution is defined.
    Nscan: int
        Number of points in the scanned y-axis on which the SQUID signal is calculated.
        Must be even.
    gamma: float
        Angle between the SQUID scan direction and the current direction.
    rho1: float
        Inner radius of the SQUID.
    rho2: float
        Outer radius of the SQUID.
    height: float
        Height of the SQUID above the device.
    phi: float
        Angle between the SQUID normal and the device plane.
    invert_normal: bool
        If True, the SQUID normal is inverted. This flips the sign of the SQUID signal.
    invert_scan: bool
        If True, the SQUID scan direction is inverted. This inverts the y-axis.

    Attributes
    ----------
    rho1: float
        Inner radius of the SQUID.
    rho2: float
        Outer radius of the SQUID.
    height: float
        Height of the SQUID above the device.
    phi: float
        Angle between the SQUID normal and the device plane.
    """

    def __init__(
        self,
        Lscan: float = 1.0,
        Wdevice: float = 0.5,
        Nscan: int = 2,
        gamma: float = 0.0,
        rho1: float = 0.1,
        rho2: float = 0.2,
        height: float = 0.2,
        phi: float = 0.0,
        invert_normal: bool = False,
        invert_scan: bool = False,
    ):
        self.rho1 = rho1
        self.rho2 = rho2
        self.height = height
        self.phi = phi

        # Set Wdevice and Nscan after setting Lscan and gamma, set them through
        # the protected setters to ensure they abide by the rules.
        self.Lscan = Lscan
        self.gamma = gamma
        self.Nscan = Nscan
        self.Wdevice = Wdevice  # Wdevice requires setting Nscan first.

        # Hyperparameters that set further orientation details:
        # 1. Is normal vector pointing up or down
        # 2. Should the scan direction be inverted to match that of the experiment
        self.invert_normal = invert_normal
        self.invert_scan = invert_scan

    # Derived parameters
    @property
    def dyscan(self) -> float:
        """Step size of the scanned y-axis.

        Returns a float representing the step size of the scanned y-axis.
        """
        return self.Lscan / self.Nscan

    @property
    def Ndevice(self) -> int:
        """Number of points in the device y-axis.

        Returns an integer representing the number of points in the device y-axis.
        """
        return int(round(self.Wdevice / self.dyproj))

    @property
    def Lproj(self) -> float:
        """Length of the projected y-axis over which the SQUID scans.

        Returns a float.
        """
        return float(self.Lscan * np.cos(self.gamma))

    @property
    def dyproj(self) -> float:
        """Step size of the projected y-axis over which the SQUID scans.

        Returns a float.
        """
        return float(self.dyscan * np.cos(self.gamma))

    # Protected setters -- these variables need to match internal rules
    # and can be adjusted by the model when set.

    # Rule: Lscan must be positive and nonzero
    @property
    def Lscan(self) -> float:
        """Length of the scanned y-axis on which the SQUID signal is calculated.

        Returns a float.
        """
        return self._Lscan

    @Lscan.setter
    def Lscan(self, value: float) -> None:
        """Set the length of the scanned y-axis.

        Ensures that Lscan is positive and nonzero, raising a ValueError if it
        is not.
        """
        if value <= 0.0:
            raise ValueError("Lscan must be positive and nonzero.")
        self._Lscan = value

    # Rule: Gamma must not be +/- pi/2
    @property
    def gamma(self) -> float:
        """Angle between the SQUID scan direction and the current direction.

        Returns a float.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value: float) -> None:
        """Set the angle between the SQUID scan direction and the current direction.

        Ensures that gamma is strictly inside (-pi/2, pi/2), raising a
        ValueError if it is not.
        """
        if not (-np.pi / 2 < value < np.pi / 2):
            raise ValueError("Gamma must be strictly inside (-pi/2, pi/2).")
        self._gamma = value

    # Rule: NScan must be even
    @property
    def Nscan(self) -> int:
        """Number of points in the scanned (data) y-axis.

        Returns an integer.
        """
        return self._Nscan

    @Nscan.setter
    def Nscan(self, value: int) -> None:
        """Set the number of points in the scanned (data) y-axis.

        Ensures that the value is even, positive, and non-zero, raising a
        ValueError if it is not.
        """
        if value <= 0:
            raise ValueError("Nscan must be positive and non-zero.")
        if value % 2 != 0:
            raise ValueError("Nscan must be an even integer.")
        self._Nscan = int(value)

    # Rule: Wdevice must be less than Lscan/cos(gamma) and be an integer
    # multiple of dyproj, and larger than zero

    @property
    def Wdevice(self) -> float:
        """Length of the device y-axis on which the current distribution is defined.

        Returns a float.
        """
        return self._Wdevice

    @Wdevice.setter
    def Wdevice(self, value: float) -> None:
        """Set the width of the device through which the current flows.

        Setter checks that the device fits fully in the scan length on both sides.
        Automatically roudns to ensure that Wdevice is an integer multiple of dyproj.
        """
        if value >= self.Lscan / np.cos(self.gamma):
            raise ValueError("Wdevice must be less than Lscan/cos(gamma).")
        if value <= 0.0:
            raise ValueError("Wdevice must be positive.")

        # Due to floating point precision, all we can do is round Wdevice
        # to the nearest multiple of dyproj.
        self._Wdevice = round(value / self.dyproj) * self.dyproj

    # Derived axes
    @property
    def yscan(self) -> torch.Tensor:
        """The scanned x-axis on which the SQUID signal is calculated."""
        return torch.linspace(-self.Lscan / 2, self.Lscan / 2 - self.dyscan, self.Nscan)

    @property
    def yproj(self) -> torch.Tensor:
        """The projected x-axis on which the SQUID signal is calculated."""
        return self.yscan * torch.tensor(np.cos(self.gamma), dtype=DTYPE)

    @property
    def ydevice(self) -> torch.Tensor:
        """The y-axis on which the current distribution is defined.

        It contains Ndevice points and has length Wdevice.
        If Ndevice is even: runs from -Wdevice/2 to Wdevice/2-dyproj. Element Ndevice/2
        is at y=0.
        If Ndevice is odd: runs from -Wdevice/2 to Wdevice/2. Element (Ndevice-1)/2 is
        at y=0.
        """
        if self.Ndevice % 2 == 0:
            return torch.linspace(
                -self.Wdevice / 2, self.Wdevice / 2 - self.dyproj, self.Ndevice
            )
        else:
            return torch.linspace(-self.Wdevice / 2, self.Wdevice / 2, self.Ndevice)

    @property
    def phi_eff(self) -> float:
        """Effective angle between the SQUID normal and the device plane.

        This calculation takes into account flipping of the scan direction.
        Returns a float.
        """
        return -self.phi if self.invert_scan else self.phi

    @property
    def kernel_sign(self) -> float:
        """Overal sign of the kernel.

        Again, this takes into account flipping of the scan direction and the
        SQUID normal. Returns a float.
        """
        sign = -1.0 if self.invert_normal else 1.0
        if self.invert_scan:
            sign = -sign
        return sign


class Forward1D:
    """Computes 1D SOT linetraces from a 1D current distribution on a device.

    The forward model for 1D current reconstruction. An instance of this
    class uses Forward1DParameters to pre-compute the SQUID-response kernel
    (Foward1D._precompute_kernel) and can then use this kernel to convert
    between a current distribution j(y'') on the device and a squid signal
    S(y) along the scan-axis.

    Parameters
    ----------
    params: Forward1DParameters
        An instance of Forward1DParameters containing the parameters for the
        forward model.
    n_alpha: int
        Number of alpha angles to use in the kernel precomputation. Defaults to 32.
    n_rho: int
        Number of rho values to use in the kernel precomputation. Defaults to 32.
    device: torch.device
        The device on which to perform all torch computations. Defaults to CPU.
    mu0: float
        Permeability of free space. Sets the units of the problem. Default is 4*pi*1e-1
        which corresponds to units of um, uA, uT.
    verbose: bool
        If True, prints progress during kernel precomputation. Defaults to False.

    Notes
    -----
    Old fitting code was unit-less. This limit can be recovered by setting mu0=4*pi,
    in which case all factors mu0/4pi disappear from the equations.
    """

    def __init__(
        self,
        params: Forward1DParameters,
        n_alpha: int = 32,
        n_rho: int = 32,
        device: torch.device | None = None,
        mu0: float = 4 * np.pi * 1e-1,
        verbose: bool = False,
    ):
        if device is None:
            device = torch.device("cpu")

        if n_alpha % 2 != 0:
            raise ValueError("n_alpha must be an even integer.")

        self._params = params
        self._n_alpha = n_alpha
        self._n_rho = n_rho
        self._device = device
        self._verbose = verbose
        self._mu0 = mu0

        self._kernel, self._kernel_fft = self._precompute_kernel()

    @property
    def params(self) -> Forward1DParameters:
        """The parameters that describe the forward model calculation.

        Returns an instance of Forward1DParameters.
        """
        return self._params

    @params.setter
    def params(self, params: Forward1DParameters) -> None:
        """Set the parameters that describe the forward model calculation.

        Upon setting, instantly precomputes all kernels again.
        """
        self._params = params
        self._kernel, self._kernel_fft = self._precompute_kernel()

    # To prevent wrapping artifacts in the convolution, we introduce the
    # following rules:
    #
    # 1. The device's y-axis is first extended to the length of the projected
    #    scan axis by padding with zeros on both sides. Then, the axis is
    #    further zero-padded with Nproj points on both sides. Final length:
    #    3*Nproj.
    # 2. The kernel is calculated on a projected y-axis of length 3*Lproj,
    #    with 3*Nproj points. To conform with FFT conventions, this y-axis
    #    runs from 0 to 3/2*Lproj, before wrapping around to -3/2*Lproj and
    #    back to 0-dyproj.

    def _ykernel(self) -> torch.Tensor:
        """Calculate the y-axis on which the kernel is defined."""
        ykernel = torch.zeros(3 * self._params.Nscan, device=self._device)
        ykernel[: int(1.5 * self._params.Nscan)] = torch.linspace(
            0,
            1.5 * self._params.Lproj - self._params.dyproj,
            int(1.5 * self._params.Nscan),
            device=self._device,
        )
        ykernel[int(1.5 * self._params.Nscan) :] = torch.linspace(
            -1.5 * self._params.Lproj,
            -self._params.dyproj,
            int(1.5 * self._params.Nscan),
            device=self._device,
        )
        return ykernel

    def _window(self, rho: float) -> float:
        """Return the window that tapers the repeated, mirrored current distribution.

        The window function:
        w(rho) = 1 for rho < rho1
                 1 - (rho-rho1)/(rho2-rho1) for rho1 <= rho < rho2
                   0 for rho >= rho2
        """
        if rho < self._params.rho1:
            return 1.0
        elif rho < self._params.rho2:
            return 1.0 - (rho - self._params.rho1) / (
                self._params.rho2 - self._params.rho1
            )
        else:
            return 0.0

    def _kernel_contribution(
        self, ykernel: torch.Tensor, alpha: float, rho: float
    ) -> torch.Tensor:
        """Calculate single kernel term (no prefactor) for given alpha and rho."""
        phi = self._params.phi_eff
        gamma = self._params.gamma
        h = self._params.height

        numerator = 2 * (
            ykernel * np.cos(phi)
            - np.cos(gamma) * np.sin(phi) * h
            - rho
            * (
                np.cos(alpha) * np.sin(gamma) * np.cos(phi)
                + np.cos(gamma) * np.sin(alpha)
            )
        )
        denominator = (
            ykernel
            - rho
            * (
                np.cos(alpha) * np.sin(gamma)
                + np.cos(gamma) * np.sin(alpha) * np.cos(phi)
            )
        ) ** 2 + (h + rho * np.sin(alpha) * np.sin(phi)) ** 2

        contribution: torch.Tensor = numerator / denominator
        return contribution

    def _precompute_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the SQUID response kernel for the current parameters."""
        n_alpha = self._n_alpha  # Alpha runs from zero to 2pi non-inclusive
        n_rho = self._n_rho  # Rho runs from 0.5*rho2/n_rho to rho2-0.5*rho2/n_rho
        # non-inclusive on both sides
        ykernel = self._ykernel()

        kernel = torch.zeros(3 * self._params.Nscan, device=self._device)

        dalpha = 2 * np.pi / n_alpha
        drho = self._params.rho2 / n_rho

        alpha_iter: Iterable[torch.Tensor] = torch.linspace(
            0.0, 2 * np.pi - dalpha, n_alpha
        )
        if self._verbose:
            print("Precomputing kernel...")
            alpha_iter = tqdm(alpha_iter, desc="Kernel (alpha)")
        rho_iter = torch.linspace(0.5 * drho, self._params.rho2 - 0.5 * drho, n_rho)
        for alpha in alpha_iter:
            for rho in rho_iter:
                w = self._window(rho)
                if w == 0.0:
                    continue

                # Calculate the contribution of this (alpha, rho) to the kernel
                y_contribution = self._kernel_contribution(ykernel, float(alpha), rho)

                # Add the contribution to the kernel, weighted by the window function
                kernel += rho * w * y_contribution * dalpha * drho

        kernel *= self._params.kernel_sign

        return kernel.to(self._device), torch.fft.fft(kernel).to(self._device)

    def forward_t(self, current: torch.Tensor) -> torch.Tensor:
        """Compute the 1D SQUID signal from a 1D current distribution on the device.

        Uses the pre-computed kernel and Fourier methods to calculate the 1D SQUID
        signal from the 1D current distribution on the device. Accelerated fully using
        Torch, which also ensures compatibility with GPU acceleration and
        autodifferentiation. For single-shot use, without autodifferentiation, it is
        easier to interface with the forward() model, which is a NumPy wrapper around
        this function.

        Parameters
        ----------
        current: torch.Tensor
            A tensor of shape (Ndevice,) representing the current distribution j(y'')
            on the device.

        Returns
        -------
        signal: torch.Tensor
            A tensor of shape (Nscan,) representing the SQUID signal on the y axis.
        """
        dyproj = self._params.dyproj

        # Step 1: Extend the current distribution to the length of the
        # projected scan axis + Nscan zero padding on both sides of the
        # projected scan axis
        current_extended = torch.zeros(3 * self._params.Nscan, device=self._device)

        start_idx = (
            self._params.Nscan + (self._params.Nscan - self._params.Ndevice) // 2
        )
        current_extended[start_idx : start_idx + self._params.Ndevice] = current

        # Step 2: Convolve with kernel using FFT
        current_fft = torch.fft.fft(current_extended)
        kernel_fft = self._kernel_fft  # Precomputed FFT of the kernel
        signal_proj_fft = current_fft * kernel_fft  # Shape: (3*Nscan,)
        # Represents SQUID signal on projected y-axis y''=cos(gamma)*y before rescaling.
        signal_proj = torch.fft.ifft(signal_proj_fft).real * dyproj

        # Step 4: keep only the center part (which is artifact free)
        # The y axis, technically, needs to be rescaled to find "signal"
        # However, we never used the y-axis, so no rescaling is neccesary
        signal: torch.Tensor = signal_proj[self._params.Nscan : 2 * self._params.Nscan]

        # Put into proper units
        signal *= self._mu0 / (4 * np.pi)

        return signal

    def forward(self, current: np.ndarray) -> np.ndarray:
        """NumPy wrapper for forward_t(). See forward_t() for details."""
        return (
            self.forward_t(
                torch.from_numpy(current).to(dtype=DTYPE, device=self._device)
            )
            .detach()
            .cpu()
            .numpy()
        )


class Forward2DParameters:
    """Parameters describing a 2D forwrad calculation from current to SOT image.

    The forward pass of the 2D model relies on parameters describing:
    - The spatial resolution and constraints of the model
    - The properties of the SQUID sensor modeled
    - The properties of the 2D underlying current distribution
    Unlike Forward1D, Forward2D has the SQUID and sample plane share a grid.

    Parameters
    ----------
    Lx: float
        Length of the scanned x-axis on which the SQUID signal is calculated.
    Ly: float
        Length of the scanned y-axis on which the SQUID signal is calculated.
    Nx: int
        Number of points in the scanned x-axis on which the SQUID signal is calculated.
        Must be even.
    Ny: int
        Number of points in the scanned y-axis on which the SQUID signal is calculated.
        Must be even. Ly/Ny must equal Lx/Nx to ensure square pixels.
    rho1: float
        Inner radius of the SQUID.
    rho2: float
        Outer radius of the SQUID.
    height: float
        Height of the SQUID above the device.
    phi: float
        Angle between the SQUID normal and the device plane.
    invert_normal: bool
        If True, the SQUID normal is inverted. This flips the sign of the SQUID signal.
    invert_scan: bool
        If True, the SQUID scan direction is inverted. This inverts the y-axis.

    Attributes
    ----------
    Lx: float
        Length of the scanned x-axis on which the SQUID signal is calculated.
    Ly: float
        Length of the scanned y-axis on which the SQUID signal is calculated.
    rho1: float
        Inner radius of the SQUID.
    rho2: float
        Outer radius of the SQUID.
    height: float
        Height of the SQUID above the device.
    phi: float
        Angle between the SQUID normal and the device plane.
    invert_normal: bool
        If True, the SQUID normal is inverted. This flips the sign of the SQUID signal.
    invert_scan: bool
        If True, the SQUID scan direction is inverted. This inverts the y-axis.
    """

    def __init__(
        self,
        Lx: float = 1.0,
        Ly: float = 1.0,
        Nx: int = 2,
        Ny: int = 2,
        rho1: float = 0.1,
        rho2: float = 0.2,
        height: float = 0.2,
        phi: float = 0.0,
        invert_normal: bool = False,
        invert_scan: bool = False,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny  # Protected property, because the pixels must be square

        self.rho1 = rho1
        self.rho2 = rho2
        self.height = height
        self.phi = phi

        # Hyperparameters that set further orientation details:
        # 1. Is normal vector pointing up or down
        # 2. Should the scan direction be inverted to match that of the experiment
        self.invert_normal = invert_normal
        self.invert_scan = invert_scan

    # Protected setters -- these variables need to match internal rules
    # and can be adjusted by the model when set.

    # Rule: Nx/Ny must be even
    @property
    def Nx(self) -> int:
        """Amount of points in the x-axis on which the SQUID signal is calculated."""
        return self._Nx

    @Nx.setter
    def Nx(self, value: int) -> None:
        """Set Nx. Ensures that Nx is even, raising a ValueError if it is not."""
        if value <= 0:
            raise ValueError("Nx must be positive and non-zero.")
        if value % 2 != 0:
            raise ValueError("Nx must be even.")
        self._Nx = value

    # Rule: pixels must be square, so Ny is set by Lx, Ly and Nx. Ny is also even.
    @property
    def Ny(self) -> int:
        """Amount of points in the y-axis on which the SQUID signal is calculated."""
        return self._Ny

    @Ny.setter
    def Ny(self, value: int) -> None:
        """Set Ny.

        Ensures that Ny is even and that pixels are square, raising a
        ValueError if not.
        """
        if value <= 0:
            raise ValueError("Ny must be positive and non-zero.")
        if value % 2 != 0:
            raise ValueError("Ny must be even.")

        dx = self.Lx / self.Nx
        dy_proposed = self.Ly / value
        if abs(dx - dy_proposed) > 1e-6:
            raise ValueError(
                "Pixels must be square, so Ny must be set such that Ly/Ny = Lx/Nx."
            )
        self._Ny = value

    # Derived axes
    @property
    def x(self) -> torch.Tensor:
        """The scanned x-axis on which the SQUID signal is calculated.

        Returns a torch.Tensor of shape (Nx,) representing the x-axis which runs from
        -L/2 to L/2-dx.
        """
        return torch.linspace(-self.Lx / 2, self.Lx / 2 - self.Lx / self.Nx, self.Nx)

    @property
    def y(self) -> torch.Tensor:
        """The projected x-axis on which the SQUID signal is calculated.

        Returns a torch.Tensor of shape (Ny,) representing the y-axis which runs from
        -L/2 to L/2-dy.
        """
        return torch.linspace(-self.Ly / 2, self.Ly / 2 - self.Ly / self.Ny, self.Ny)

    @property
    def xx(self) -> torch.Tensor:
        """The x-axis (grid) on which the SQUID signal is calculated.

        Returns a torch.Tensor of shape (Nx, Ny) representing the x-coordinate grid.
        """
        x = self.x
        y = self.y
        xx, _ = torch.meshgrid(x, y, indexing="ij")
        return xx

    @property
    def yy(self) -> torch.Tensor:
        """The y-axis (grid) on which the SQUID signal is calculated.

        Returns a torch.Tensor of shape (Nx, Ny) representing the y-coordinate grid.
        """
        x = self.x
        y = self.y
        _, yy = torch.meshgrid(x, y, indexing="ij")
        return yy

    @property
    def phi_eff(self) -> float:
        """Effective tilt: invert_scan works in the scan frame, phi -> -phi."""
        return -self.phi if self.invert_scan else self.phi

    @property
    def kernel1_sign(self) -> float:
        """Sign of the Jx-coupling kernel: K1(x,-y;-phi) = -K1(x,y;phi)."""
        sign = -1.0 if self.invert_normal else 1.0
        return -sign if self.invert_scan else sign

    @property
    def kernel2_sign(self) -> float:
        """Sign of the Jy-coupling kernel: K2(x,-y;-phi) = +K2(x,y;phi)."""
        return -1.0 if self.invert_normal else 1.0


class Forward2DCurrent:
    """Forward calculation of 2D SQUID images from 2D current (Jx/Jy) distributions.

    The forward model for 2D current reconstruction. An instance of this
    class uses Forward2DParameters to pre-compute the SQUID-response kernel
    (Foward2DCurrent._precompute_kernel1 and Foward2DCurrent._precompute_kernel2) and
    then uses this kernel to convert between a current distribution Jx/Jy on the device
    and a SQUID signal S on the same grid.

    The implementation contains an approach to minimize wrapping artifacts in the
    convolution:
    1. All convolutions are performed on a grid of size 3*Nx by 3*Ny to ensure
    sufficient padding protects against wrapping artifacts.
    2. The current distributions are mirrored and extended over the entire 3*Nx by 3*Ny
    grid to simulate continuity.
    3. Outside the main FOV, the current distributions are tapered to zero using a
    cos(pi*x) window over half Nx/Ny to prevent faraway imaginary currents from
    contributing to the SQUID signal.

    Parameters
    ----------
    params: Forward2DParameters
        An instance of Forward2DParameters containing the parameters for the
        forward model.
    n_alpha: int
        Number of alpha angles to use in the kernel precomputation. Defaults to 32.
    n_rho: int
        Number of rho values to use in the kernel precomputation. Defaults to 32.
    device: torch.device
        The device on which to perform all torch computations. Defaults to CPU.
    mu0: float
        Permeability of free space. Sets the units of the problem. Default is 4*pi*1e-1
        which corresponds to units of um, uA, uT.
    verbose: bool
        If True, prints progress during kernel precomputation. Defaults to False.

    Notes
    -----
    Old fitting code was unit-less. This limit can be recovered by setting mu0=4*pi,
    in which case all factors mu0/4pi disappear from the equations.
    """

    def __init__(
        self,
        params: Forward2DParameters,
        n_alpha: int = 32,
        n_rho: int = 32,
        device: torch.device | None = None,
        mu0: float = 4 * np.pi * 1e-1,
        verbose: bool = False,
    ):
        if device is None:
            device = torch.device("cpu")

        self._params = params
        self._n_alpha = n_alpha
        self._n_rho = n_rho
        self._device = device
        self._mu0 = mu0
        self._verbose = verbose

        self._kernel1, self._kernel1_fft = self._precompute_kernel1()
        self._kernel2, self._kernel2_fft = self._precompute_kernel2()
        self._window = self._precompute_window()
        self._xmap, self._ymap = self._precompute_reflect_maps()

    @property
    def params(self) -> Forward2DParameters:
        """The parameters that describe the forward model calculation."""
        return self._params

    @params.setter
    def params(self, params: Forward2DParameters) -> None:
        """Set the parameters that describe the forward model calculation.

        After setting, instantly precomputes all kernels again as well as windows
        and reflection maps.
        """
        self._params = params
        self._kernel1, self._kernel1_fft = self._precompute_kernel1()
        self._kernel2, self._kernel2_fft = self._precompute_kernel2()
        self._window = self._precompute_window()
        self._xmap, self._ymap = self._precompute_reflect_maps()

    def _gridkernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the kernel xx/yy axes on the 3*Nx by 3*Ny grid."""
        xkernel = torch.zeros(3 * self._params.Nx)
        xkernel[: int(1.5 * self._params.Nx)] = torch.linspace(
            0,
            1.5 * self._params.Lx - self._params.Lx / self._params.Nx,
            int(1.5 * self._params.Nx),
        )
        xkernel[int(1.5 * self._params.Nx) :] = torch.linspace(
            -1.5 * self._params.Lx,
            -self._params.Lx / self._params.Nx,
            int(1.5 * self._params.Nx),
        )

        ykernel = torch.zeros(3 * self._params.Ny)
        ykernel[: int(1.5 * self._params.Ny)] = torch.linspace(
            0,
            1.5 * self._params.Ly - self._params.Ly / self._params.Ny,
            int(1.5 * self._params.Ny),
        )
        ykernel[int(1.5 * self._params.Ny) :] = torch.linspace(
            -1.5 * self._params.Ly,
            -self._params.Ly / self._params.Ny,
            int(1.5 * self._params.Ny),
        )

        xxkernel, yykernel = torch.meshgrid(xkernel, ykernel, indexing="ij")

        return xxkernel, yykernel

    def _kernel1_contribution(
        self, xxkernel: torch.Tensor, yykernel: torch.Tensor, alpha: float, rho: float
    ) -> torch.Tensor:
        phi = self._params.phi_eff
        h = self._params.height

        px = xxkernel + rho * math.cos(alpha)
        py = yykernel - rho * math.sin(alpha) * math.cos(phi)
        pz = h + rho * math.sin(alpha) * math.sin(phi)
        p3 = (px**2 + py**2 + pz**2) ** 1.5

        return (math.cos(phi) * py - math.sin(phi) * pz) / p3

    def _kernel2_contribution(
        self, xxkernel: torch.Tensor, yykernel: torch.Tensor, alpha: float, rho: float
    ) -> torch.Tensor:
        phi = self._params.phi_eff
        h = self._params.height

        px = xxkernel + rho * math.cos(alpha)
        py = yykernel - rho * math.sin(alpha) * math.cos(phi)
        pz = h + rho * math.sin(alpha) * math.sin(phi)
        p3 = (px**2 + py**2 + pz**2) ** 1.5

        return -math.cos(phi) * px / p3

    def _precompute_kernel1(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the SQUID response kernel for the current parameters."""
        n_alpha = self._n_alpha  # Alpha runs from zero to 2pi non-inclusive
        n_rho = self._n_rho  # Rho runs from 0.5*rho2/n_rho to rho2-0.5*rho2/n_rho
        # non-inclusive on both sides
        xxkernel, yykernel = self._gridkernel()

        kernel = torch.zeros((3 * self._params.Nx, 3 * self._params.Ny))

        dalpha = 2 * np.pi / n_alpha
        drho = self._params.rho2 / n_rho

        alpha_iter: Iterable[torch.Tensor] = torch.linspace(
            0.0, 2 * np.pi - dalpha, n_alpha
        )
        if self._verbose:
            print("Precomputing kernel...")
            alpha_iter = tqdm(alpha_iter, desc="Kernel (alpha)")
        rho_iter = torch.linspace(0.5 * drho, self._params.rho2 - 0.5 * drho, n_rho)
        for alpha in alpha_iter:
            for rho in rho_iter:
                w = self._windowfunc(float(rho))
                if w == 0.0:
                    continue

                # Calculate the contribution of this (alpha, rho) to the kernel
                contribution = self._kernel1_contribution(
                    xxkernel, yykernel, float(alpha), rho
                )

                # Add the contribution to the kernel, weighted by the window function
                kernel += rho * w * contribution * dalpha * drho

        kernel = kernel * self._params.kernel1_sign

        return kernel.to(self._device), torch.fft.fft2(kernel).to(self._device)

    def _precompute_kernel2(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the SQUID response kernel for the current parameters."""
        n_alpha = self._n_alpha  # Alpha runs from zero to 2pi non-inclusive
        n_rho = self._n_rho  # Rho runs from 0.5*rho2/n_rho to rho2-0.5*rho2/n_rho
        # non-inclusive on both sides
        xxkernel, yykernel = self._gridkernel()

        kernel = torch.zeros((3 * self._params.Nx, 3 * self._params.Ny))

        dalpha = 2 * np.pi / n_alpha
        drho = self._params.rho2 / n_rho

        alpha_iter: Iterable[torch.Tensor] = torch.linspace(
            0.0, 2 * np.pi - dalpha, n_alpha
        )
        if self._verbose:
            print("Precomputing kernel...")
            alpha_iter = tqdm(alpha_iter, desc="Kernel (alpha)")
        rho_iter = torch.linspace(0.5 * drho, self._params.rho2 - 0.5 * drho, n_rho)
        for alpha in alpha_iter:
            for rho in rho_iter:
                w = self._windowfunc(rho)
                if w == 0.0:
                    continue

                # Calculate the contribution of this (alpha, rho) to the kernel
                contribution = self._kernel2_contribution(
                    xxkernel, yykernel, float(alpha), rho
                )

                # Add the contribution to the kernel, weighted by the window function
                kernel += rho * w * contribution * dalpha * drho

        kernel = kernel * self._params.kernel2_sign

        return kernel.to(self._device), torch.fft.fft2(kernel).to(self._device)

    def _precompute_window(self) -> torch.Tensor:
        def window1d(N: int) -> torch.Tensor:
            halfN = N // 2
            w = torch.zeros(3 * N, device=self._device)
            ramp = torch.cos(
                np.pi / 2 * torch.linspace(-1, 0, halfN, device=self._device)
            )
            w[N - halfN : N] = ramp
            w[2 * N : 2 * N + halfN] = torch.flip(ramp, dims=[0])
            w[N : 2 * N] = 1.0
            return w

        w_x_1d = window1d(self._params.Nx)
        w_y_1d = window1d(self._params.Ny)
        window: torch.Tensor = w_x_1d[:, None] * w_y_1d[None, :]
        return window

    def _precompute_reflect_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        Nx = self._params.Nx
        Ny = self._params.Ny

        ix = torch.arange(3 * Nx, device=self._device)
        iy = torch.arange(3 * Ny, device=self._device)

        xmap = torch.where(
            ix < Nx, Nx - ix - 1, torch.where(ix < 2 * Nx, ix - Nx, 3 * Nx - ix - 1)
        )
        ymap = torch.where(
            iy < Ny, Ny - iy - 1, torch.where(iy < 2 * Ny, iy - Ny, 3 * Ny - iy - 1)
        )
        return xmap, ymap

    def _extend_current(self, current: torch.Tensor) -> torch.Tensor:
        return current[self._xmap[:, None], self._ymap[None, :]] * self._window

    def _windowfunc(self, rho: float) -> float:
        """Return the window that tapers the repeated, mirrored current distribution.

        The window function:
        w(rho) = 1 for rho < rho1
                 1 - (rho-rho1)/(rho2-rho1) for rho1 <= rho < rho2
                   0 for rho >= rho2
        """
        if rho < self._params.rho1:
            return 1.0
        elif rho < self._params.rho2:
            return 1.0 - (rho - self._params.rho1) / (
                self._params.rho2 - self._params.rho1
            )
        else:
            return 0.0

    def forward_t(
        self, current_x: torch.Tensor, current_y: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the 2D SQUID signal from a 2D current distribution on the device.

        Uses the pre-computed kernels and Fourier methods to calculate the 2D SQUID
        signal from the 2D current distribution on the device. Accelerated fully using
        Torch, which also ensures compatibility with GPU acceleration and
        autodifferentiation. For single-shot use, without autodifferentiation, it is
        easier to interface with the forward() model, which is a NumPy wrapper around
        this function.

        Parameters
        ----------
        current_x: torch.Tensor
            A tensor of shape (Nx, Ny) representing the x component of the current
            distribution j(x,y)
        current_y: torch.Tensor
            A tensor of shape (Nx, Ny) representing the y component of the current
            distribution j(x,y)

        Returns
        -------
        signal: torch.Tensor
            A tensor of shape (Nx, Ny) representing the SQUID signal on the x and y
            axes.
        """
        dx = self._params.Lx / self._params.Nx
        dy = dx  # This is a contract!

        # Step 1: Pad the current distribution to (3*Nx, 3*Ny) with mirroring and
        # tapering
        current_x_extended = self._extend_current(current_x)
        current_y_extended = self._extend_current(current_y)

        # Step 2: Convolve with kernels using FFT
        current_x_fft = torch.fft.fft2(current_x_extended)
        current_y_fft = torch.fft.fft2(current_y_extended)
        kernel1_fft = self._kernel1_fft  # Precomputed FFT of the kernel
        kernel2_fft = self._kernel2_fft  # Precomputed FFT of the kernel
        signal_proj_fft = current_x_fft * kernel1_fft + current_y_fft * kernel2_fft
        signal_proj = torch.fft.ifft2(signal_proj_fft).real * dx * dy

        # Step 4: keep only the center part (which is artifact free)
        signal: torch.Tensor = signal_proj[
            self._params.Nx : 2 * self._params.Nx, self._params.Ny : 2 * self._params.Ny
        ]

        # Put into proper units
        signal *= self._mu0 / (4 * np.pi)

        return signal

    def forward(self, current_x: np.ndarray, current_y: np.ndarray) -> np.ndarray:
        """NumPy wrapper for forward_t(). See forward_t() for details."""
        return (
            self.forward_t(
                torch.from_numpy(current_x).to(dtype=DTYPE, device=self._device),
                torch.from_numpy(current_y).to(dtype=DTYPE, device=self._device),
            )
            .detach()
            .cpu()
            .numpy()
        )


class MeshToGrid:
    """Converts between a streamfunction on a mesh and a current distribution on a grid.

    Contains the full toolchain for converting a streamfucntion on a mesh to a current
    on a grid. This is what is required to perform a forward chain from an on-mesh
    streamfunction to a SQUID image. On-mesh streamfunctions are preferred over on-grid
    currents because they are guaranteed divergence free and guaranteed fit the boundary
    conditions of the problem.

    Parameters
    ----------
    mesh: SimplyConnectedCurrentMesh
        The mesh on which the streamfunction is defined.
    forwardparams: Forward2DParameters
        The parameters describing the grid on which the current is defined.
    device: torch.device
        The device on which to perform all torch computations. Defaults to CPU.
    """

    def __init__(
        self,
        mesh: SimplyConnectedCurrentMesh,
        forwardparams: Forward2DParameters,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")

        self._mesh = mesh
        self._params = forwardparams
        self._device = device

        # Our goal now is to construct matrices that will take the streamfunction on the
        # mesh to current
        # densities Jx, Jy on a grid.
        # Step 1 is to go from streamfunction to current on the mesh.
        triangles = self._mesh.triangles
        shapegradients = self._mesh.shapegradients
        rows = np.repeat(np.arange(len(triangles)), 3)
        cols = triangles.ravel()
        vals_x = shapegradients[:, :, 1].ravel()  # Jx = d(psi)/dy
        vals_y = -shapegradients[:, :, 0].ravel()  # Jy = -d(psi)/dx
        self._mesh_to_current_x = sparse.coo_matrix(
            (vals_x, (rows, cols)), shape=(len(triangles), len(self._mesh.vertices))
        ).tocsr()
        self._mesh_to_current_y = sparse.coo_matrix(
            (vals_y, (rows, cols)), shape=(len(triangles), len(self._mesh.vertices))
        ).tocsr()

        # Step 2 is to go from the cell-centered Jx/Jy = mesh_to_current_i @ psi to a
        # grid of Jx/Jy on the axes defined by forwardparams.

        xx, yy = self._params.xx, self._params.yy
        grid_points = torch.stack([xx.ravel(), yy.ravel()], dim=1).cpu().numpy()
        tri_obj = mtri.Triangulation(
            mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles
        )
        tri_idx: np.ndarray = np.asarray(
            tri_obj.get_trifinder()(grid_points[:, 0], grid_points[:, 1])
        )  # (N_grid,) int, -1 if outside
        inside = tri_idx >= 0
        inside_idx = np.where(inside)[0]
        rows = inside_idx
        cols = tri_idx[inside]
        vals = np.ones(inside.sum(), dtype=np.float64)

        self._grid_to_mesh = sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(grid_points), len(self._mesh.triangles))
        ).tocsr()

        # These matrices can be combined to two matrices Mx and My that take
        # streamfunction to Jx and Jy respectively
        Mx_scipy = self._grid_to_mesh @ self._mesh_to_current_x
        My_scipy = self._grid_to_mesh @ self._mesh_to_current_y

        # Store as sparse torch for use in the Torch chains
        Mx_coo = Mx_scipy.tocoo()
        My_coo = My_scipy.tocoo()
        indices_x = torch.stack(
            [
                torch.from_numpy(Mx_coo.row.astype(np.int64)),
                torch.from_numpy(Mx_coo.col.astype(np.int64)),
            ],
            dim=0,
        )
        indices_y = torch.stack(
            [
                torch.from_numpy(My_coo.row.astype(np.int64)),
                torch.from_numpy(My_coo.col.astype(np.int64)),
            ],
            dim=0,
        )
        self._Mx = (
            torch.sparse_coo_tensor(
                indices_x,
                torch.from_numpy(Mx_coo.data).to(DTYPE),
                size=Mx_coo.shape,
                device=self._device,
            )
            .coalesce()
            .to(self._device)
        )
        self._My = (
            torch.sparse_coo_tensor(
                indices_y,
                torch.from_numpy(My_coo.data).to(DTYPE),
                size=My_coo.shape,
                device=self._device,
            )
            .coalesce()
            .to(self._device)
        )

    def streamfunction_to_currents_t(
        self,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate current densities on a grid from a streamfunction on a mesh.

        Takes a streamfunction g defined on the mesh vertices and returns the current
        densities Jx, Jy on the grid. The grid points are taken from the forward
        parameters passed at construction. This function is fully Torch compatible and
        can be used with GPU acceleration and autodifferentiation. For single-shot use,
        without autodifferentiation, it is easier to interface with
        streamfunction_to_currents(), which is a NumPy wrapper around this function.

        Parameters
        ----------
        g: torch.Tensor
            A tensor of shape (N_vertices,) representing the streamfunction on the mesh
            vertices.

        Returns
        -------
        Jx_grid: torch.Tensor
            A tensor of shape (Nx, Ny) representing the x component of the current
            distribution on the grid.
        Jy_grid: torch.Tensor
            A tensor of shape (Nx, Ny) representing the y component of the current
            distribution on the grid.
        """
        g = g.to(device=self._device, dtype=self._Mx.dtype)
        Jx_grid = (self._Mx @ g).reshape(self._params.Nx, self._params.Ny)
        Jy_grid = (self._My @ g).reshape(self._params.Nx, self._params.Ny)
        return Jx_grid, Jy_grid

    def streamfunction_to_currents(
        self,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy wrapper for streamfunction_to_currents_t().

        See streamfunction_to_currents_t() for details.
        """
        g_torch = torch.from_numpy(g).to(device=self._device, dtype=self._Mx.dtype)
        Jx_grid, Jy_grid = self.streamfunction_to_currents_t(g_torch)
        return Jx_grid.detach().cpu().numpy(), Jy_grid.detach().cpu().numpy()

    @property
    def mesh(self) -> SimplyConnectedCurrentMesh:
        """The mesh on which the streamfunction is defined."""
        return self._mesh
