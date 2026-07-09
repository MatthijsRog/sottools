import numpy as np
from matplotlib import pyplot as plt

from sottools.forward import Forward1D, Forward2DCurrent


def plot_1dforwardkernel(
    forward: Forward1D,
    ax: plt.Axes,
):
    """Plot the precomputed kernel of a Forward1D object.

    This kernel is used to translate current to SQUID signal, incorporating Biot-Savart
    physics as well as SQUID geometry.

    Parameters
    ----------
    forward: Forward1D
        The Forward1D object containing the precomputed kernel.
    ax: plt.Axes
        The matplotlib axes to plot on. Mutable.
    """
    ykernel = forward._ykernel().cpu().numpy()
    kernel = forward._kernel.cpu().numpy()

    # Sort back
    ykernel_sorted_indices = np.argsort(ykernel)
    ykernel_sorted = ykernel[ykernel_sorted_indices]
    kernel_sorted = kernel[ykernel_sorted_indices]

    ax.plot(ykernel_sorted, kernel_sorted, label="Precomputed Kernel")
    ax.set_xlabel("Projected y-axis (ykernel)")
    ax.set_ylabel("Kernel Value")


def plot_2dforwardkernels(forward: Forward2DCurrent, axes: [plt.Axes, plt.Axes]):
    """Plot the precomputed kernels of a Forward2DCurrent object.

    These kernels are used to translate current to SQUID signal, incorporating
    Biot-Savart physics as well as SQUID geometry.

    Parameters
    ----------
    forward: Forward2DCurrent
        The Forward2DCurrent object containing the precomputed kernels.
    axes: list of plt.Axes
        The matplotlib axes to plot on. Must be a list of two axes.
    """
    xxkernel, yykernel = forward._gridkernel()
    kernel1 = forward._kernel1.cpu().numpy()
    kernel2 = forward._kernel2.cpu().numpy()

    kernel1 = np.fft.ifftshift(kernel1)
    kernel2 = np.fft.ifftshift(kernel2)

    fig = axes[0].get_figure()
    im1 = axes[0].imshow(
        kernel1,
        extent=(yykernel.min(), yykernel.max(), xxkernel.max(), xxkernel.min()),
        origin="upper",
    )
    axes[0].set_title("Precomputed Kernel 1 (Jx contribution)")
    axes[0].set_xlabel("Y")
    axes[0].set_ylabel("X")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        kernel2,
        extent=(yykernel.min(), yykernel.max(), xxkernel.max(), xxkernel.min()),
        origin="upper",
    )
    axes[1].set_title("Precomputed Kernel 2 (Jy contribution)")
    axes[1].set_xlabel("Y")
    axes[1].set_ylabel("X")
    fig.colorbar(im2, ax=axes[1])
