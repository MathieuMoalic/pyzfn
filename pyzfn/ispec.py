import numpy as np
from collections import namedtuple
from typing import Any, TYPE_CHECKING
from matplotlib import pyplot as plt
from matplotlib.axes import Axes, SubplotBase
from matplotlib.backend_bases import MouseEvent, Event, MouseButton
from numpy.typing import NDArray

from .utils import indexes

if TYPE_CHECKING:
    from pyzfn import Pyzfn

# Use this only for documentation, not isinstance checks
axType = NDArray[Any] | SubplotBase | Axes


def inner_ispec(
    self: "Pyzfn",
    dset_str: str = "m",
    thres: float = 0.1,
    min_dist: int = 5,
    fmin: float = 0,
    fmax: float = 40,
    c: int = 0,
    log: bool = False,
    z: int = 0,
) -> None:
    Peak = namedtuple("Peak", ["idx", "freq", "amp"])

    dx, dy = self.attrs["dx"], self.attrs["dy"]
    if not isinstance(dx, float) or not isinstance(dy, float):
        raise ValueError("dx and dy must be floats")

    def get_peaks(x: NDArray[np.float64], y: NDArray[np.float64]) -> list[Peak]:
        idx = indexes(y, thres=thres, min_dist=min_dist)
        peak_amp = [y[i] for i in idx]
        freqs = [x[i] for i in idx]
        return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]

    def plot_spectra(
        ax: Axes, x: NDArray[np.float64], y: NDArray[np.float64], peaks: list[Peak]
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
        ax.set_title(self.name)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def plot_modes(axes: Any, f: float) -> None:
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        for ax in axes.flatten():
            ax.cla()
            ax.set(xticks=[], yticks=[])
        mode = self.get_mode(dset_str, f)[z]
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
    peaks = get_peaks(x, y)
    plot_spectra(ax_spec, x, y, peaks)
    axes_modes = gs[0, 1].subgridspec(3, 3).subplots()
    vline = ax_spec.axvline((fmax + fmin) / 2, ls="--", lw=0.8, c="#ffb86c")

    def onclick(event: Event) -> None:
        if isinstance(event, MouseEvent) and event.inaxes == ax_spec:
            f: float = (fmax + fmin) / 2
            if event.button == MouseButton.RIGHT:
                freqs = np.array([p.freq for p in peaks])
                f = freqs[np.abs(freqs - event.xdata).argmin()]
            else:
                xdata = event.xdata
                if not isinstance(xdata, float):
                    return
                f = xdata
            vline.set_data([f, f], [0, 1])
            plot_modes(axes_modes, f)
            fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
