"""Interactive spectrum plotting and mode visualization utilities for Pyzfn.

This module provides functions to plot spectra, identify peaks, and interactively
explore mode shapes using matplotlib.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseButton, MouseEvent
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .utils import Peak, find_peaks

if TYPE_CHECKING:  # pragma: no cover
    from pyzfn import Pyzfn

AxesArray = NDArray[np.object_]


def _plot_spectra(
    ax: Axes,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    peaks: list[Peak],
) -> None:
    ax.plot(x, y)
    for _, freq, amp in peaks:
        ax.text(
            freq,
            amp + 0.03 * float(np.max(y)),
            f"{freq:.2f}",
            rotation=90,
            ha="center",
            va="bottom",
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_modes(
    axes: AxesArray,
    mode: NDArray[np.complex64],
    dx: float,
    dy: float,
) -> None:
    for ax in axes.flatten():
        ax.cla()
        ax.set(xticks=[], yticks=[])

    extent = [
        0,
        mode.shape[1] * dx * 1e9,
        0,
        mode.shape[0] * dy * 1e9,
    ]
    for i in range(3):
        abs_arr = np.abs(mode[:, :, i])
        phase_arr = np.angle(mode[:, :, i])
        axes[0, i].imshow(
            abs_arr,
            cmap="inferno",
            vmin=0,
            vmax=float(abs_arr.max()),
            extent=extent,
            interpolation="none",
            aspect="equal",
        )
        axes[1, i].imshow(
            phase_arr,
            aspect="equal",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="none",
            extent=extent,
        )
        axes[2, i].imshow(
            phase_arr,
            aspect="equal",
            alpha=abs_arr / abs_arr.max(),
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
            extent=extent,
        )


def inner_ispec(  # noqa: PLR0913
    self: "Pyzfn",
    dset_str: str = "m",
    *,
    thres: float = 0.1,
    min_dist: int = 5,
    fmin: float = 0,
    fmax: float = 40,
    c: int = 0,
    log: bool = False,
    z: int = 0,
) -> tuple[Callable[[Event], None], Figure, Axes, AxesArray]:
    """Interactively plot the spectrum and visualize mode shapes for a given dataset.

    Parameters
    ----------
    self : Pyzfn
        The Pyzfn instance containing data and methods.
    dset_str : str, optional
        Dataset string identifier (default is "m").
    thres : float, optional
        Threshold for peak finding (default is 0.1).
    min_dist : int, optional
        Minimum distance between peaks (default is 5).
    fmin : float, optional
        Minimum frequency to display (default is 0).
    fmax : float, optional
        Maximum frequency to display (default is 40).
    c : int, optional
        Channel index (default is 0).
    log : bool, optional
        Whether to plot the log of the spectrum (default is False).
    z : int, optional
        Mode index (default is 0).

    Returns
    -------
    tuple
        (onclick handler, Figure, Axes for spectrum, AxesArray for modes)

    Raises
    ------
    TypeError
        If `dx` or `dy` are not floats.

    """
    dx, dy = self.attrs["dx"], self.attrs["dy"]
    if not isinstance(dx, float) or not isinstance(dy, float):
        msg = "dx and dy must be floats"
        raise TypeError(msg)

    def get_spectrum() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_arr = self.get_array(f"fft/{dset_str}/freqs")
        y_arr = self.get_array(f"fft/{dset_str}/spec")[:, c]
        x = np.asarray(x_arr, dtype=np.float64)
        y = np.asarray(y_arr, dtype=np.float64)
        if log:
            y = np.log(y)
            y -= y.min()
            y /= y.max()
        x1 = np.abs(x - fmin).argmin()
        x2 = np.abs(x - fmax).argmin()
        return x[x1:x2], y[x1:x2]

    fig = plt.figure(constrained_layout=True, figsize=(15, 6))
    gs = fig.add_gridspec(1, 2)
    ax_spec = fig.add_subplot(gs[0, 0])
    x, y = get_spectrum()
    peaks = find_peaks(x, y, thres=thres, min_dist=min_dist)
    _plot_spectra(ax_spec, x, y, peaks)
    axes_modes = cast("AxesArray", gs[0, 1].subgridspec(3, 3).subplots())
    vline = ax_spec.axvline((fmax + fmin) / 2, ls="--", lw=0.8, c="#ffb86c")

    def onclick(event: Event) -> None:
        if isinstance(event, MouseEvent) and event.inaxes == ax_spec:
            f: float = (fmax + fmin) / 2
            if not isinstance(event.xdata, float):
                return
            if event.button == MouseButton.RIGHT:
                freqs = np.array([p.frequency for p in peaks])
                f = freqs[np.abs(freqs - event.xdata).argmin()]
            else:
                f = float(event.xdata)
            vline.set_data([f, f], [0, 1])
            mode = self.get_mode(dset_str, f)[z]
            _plot_modes(axes_modes, mode, dx, dy)
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    return onclick, fig, ax_spec, axes_modes
