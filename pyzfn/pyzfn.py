import os
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import zarr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes, SubplotBase
from matplotlib.backend_bases import Event, MouseEvent
from numpy.typing import NDArray
from tqdm import trange


from .utils import (
    check_memory,
    get_closest_point_on_fig,
    hsl2rgb,
    indexes,
)

npf32 = NDArray[Any]
npc64 = NDArray[Any]

axType = NDArray[Any] | SubplotBase | Axes
SliceElement = Union[int, slice, None, type(Ellipsis)]
ArraySlice = Union[SliceElement, Tuple[SliceElement, ...]]


class Pyzfn:
    """Postprocessing, visualization for amumax's zarr outputs"""

    def __init__(self, path: str) -> None:
        warnings.filterwarnings(
            "ignore",
            message="Object at .* is not recognized as a component of a Zarr hierarchy.",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path Not Found : '{path}'")

        self.z = zarr.open_group(path, mode="a")
        self.__dict__.update(self.z.__dict__)

        self.path: Path = Path(path).absolute()
        self.name: str = self.path.name.replace(self.path.suffix, "")

    def __repr__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __str__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __name__(self) -> str:
        return self.name

    @property
    def pp(self) -> Any:
        """Pretty print the tree"""
        return self.z.tree(expand=True)

    @property
    def p(self) -> None:
        """Print the tree"""
        print(self.name, self.z.tree())

    @property
    def dx(self) -> float:
        dx = self.z.attrs["dx"]
        if not isinstance(dx, float):
            raise ValueError("dx must be a float")
        return dx

    @property
    def dy(self) -> float:
        dy = self.z.attrs["dy"]
        if not isinstance(dy, float):
            raise ValueError("dy must be a float")
        return dy

    @property
    def dz(self) -> float:
        dz = self.z.attrs["dz"]
        if not isinstance(dz, float):
            raise ValueError("dz must be a float")
        return dz

    def get_dset(self, dset: str) -> zarr.Array:
        dset_tmp = self.z[dset]
        if isinstance(dset_tmp, zarr.Group):
            raise ValueError(f"`{dset}` is a group, not a dataset.")
        return dset_tmp

    def calc_disp(
        self, dset_in_str: str = "m", dset_out_str: str = "m", single_y: bool = False
    ) -> None:
        dset_in = self.get_dset(dset_in_str)
        dset_out: str = f"disp/{dset_out_str}"
        self.rm(dset_out)
        s = dset_in.shape
        arr_out = np.zeros((s[0] // 2 - 1, s[3] - 1, s[4]), dtype=np.float32)
        hann2d = np.sqrt(np.outer(np.hanning(s[0] - 1), np.hanning(s[3] - 1)))[
            :, :, None
        ]
        arr = np.empty((s[0] - 1, s[3] - 1, s[4]), dtype=np.complex64)

        ychunks = dset_in.chunks[2]
        for iy in trange(
            dset_in.shape[2] // ychunks - 1,
            leave=False,
            desc=f"Calculating the dispersion for `{self.name}`",
        ):
            arr[:] = np.sum(
                dset_in[1:, 0, iy * ychunks : (iy + 1) * ychunks, 1:, :], axis=2
            )
            arr *= hann2d
            out = np.fft.fftn(arr, axes=(1, 0))
            out -= np.average(out, axis=(0, 1), keepdims=True)
            out = out[: out.shape[0] // 2]
            out = np.fft.fftshift(out, axes=(1, 2))
            arr_out[:] += np.abs(out[: arr_out.shape[0]])
            if single_y:
                break

        self.z.create_dataset(f"{dset_out}/arr", data=arr_out)
        ts = np.asarray(dset_in.attrs["t"])
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        self.z.create_dataset(f"{dset_out}/freqs", data=freqs[1:], chunks=False)
        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.dx)) * 2 * np.pi
        self.z.create_dataset(f"{dset_out}/kvecs", data=kvecs, chunks=None)

    def rm(self, dset: str) -> None:
        del self.z[dset]

    def calc_modes(
        self,
        dset_in_str: str = "m",
        dset_out_str: str = "m",
        window: bool = True,
        tmin: int = 0,
        tmax: Optional[int] = None,
        zmin: int = 0,
        zmax: Optional[int] = None,
        ymin: int = 0,
        ymax: Optional[int] = None,
        xmin: int = 0,
        xmax: Optional[int] = None,
        cmin: int = 0,
        cmax: Optional[int] = None,
        skip_memory_check: bool = False,
    ) -> None:
        """
        Compute FFT modes over a Zarr dataset and save results into the store.

        Parameters:
            dset_in_str: Input dataset key in `self.z`.
            dset_out_str: Output dataset key prefix for storing FFT results.
            window: Whether to apply a Hanning window in time.
            tmin/tmax, zmin/zmax, ymin/ymax, xmin/xmax, cmin/cmax: Bounds for slicing.
            skip_memory_check: If True, skip memory checks.
        """
        dset_in = self.z[dset_in_str]
        full_shape = tuple(int(x) for x in dset_in.shape)
        tmax = tmax if tmax is not None else full_shape[0]
        zmax = zmax if zmax is not None else full_shape[1]
        ymax = ymax if ymax is not None else full_shape[2]
        xmax = xmax if xmax is not None else full_shape[3]
        cmax = cmax if cmax is not None else full_shape[4]

        slices = [
            slice(tmin, tmax),
            slice(zmin, zmax),
            slice(ymin, ymax),
            slice(xmin, xmax),
            slice(cmin, cmax),
        ]
        print(
            f"Calculating modes for {dset_in_str} with shape {full_shape} "
            f"and slices {slices}"
        )
        # Perform memory check
        msg = check_memory([tuple(slices)], full_shape, force=skip_memory_check)
        if msg is not None:
            print(msg)
        ts = dset_in.attrs["t"]
        if not isinstance(ts, list):
            raise ValueError("t must be a list")
        ts = np.array(dset_in.attrs["t"])[:tmax]

        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        self.z.create_dataset(
            f"fft/{dset_out_str}/freqs", data=freqs, chunks=False, overwrite=True
        )
        self.z.create_dataset(
            f"modes/{dset_out_str}/freqs", data=freqs, chunks=False, overwrite=True
        )
        arr = np.asarray(dset_in[tmin:tmax, zmin:zmax, ymin:ymax, xmin:xmax, cmin:cmax])
        arr -= arr.mean(axis=0)[None, ...]
        if window:
            arr *= np.hanning(arr.shape[0])[:, None, None, None, None]
        arr = np.fft.rfft(arr, axis=0)
        self.z.create_dataset(
            f"modes/{dset_out_str}/arr",
            data=arr,
            dtype=np.complex64,
            chunks=(1, None, None, None, None),
        )
        arr = np.abs(arr)
        self.z.create_dataset(
            f"fft/{dset_out_str}/spec",
            chunks=False,
            data=np.max(arr, axis=(1, 2, 3)),
            overwrite=True,
        )
        self.z.create_dataset(
            f"fft/{dset_out_str}/sum",
            chunks=False,
            data=np.sum(arr, axis=(1, 2, 3)),
            overwrite=True,
        )

    def get_mode(self, dset: str, f: float, c: Union[int, None] = None):
        real_dset = ""
        if f"modes/{dset}/arr" in self.z:
            real_dset = f"modes/{dset}"
        elif f"tmodes/{dset}/arr" in self.z:
            real_dset = f"tmodes/{dset}"
        else:
            raise ValueError("`modes` or `tmodes` not found.")
        ds1 = self.z[f"{real_dset}/freqs"]
        arr1 = np.asarray(ds1[:], dtype=np.float32)
        fi = int((np.abs(arr1 - f)).argmin())
        arr = self.z[f"{real_dset}/arr"][
            (fi, slice(None), slice(None), slice(None), slice(None))
        ]
        if c is None:
            return arr
        return arr[..., c]

    def ispec(
        self,
        dset: str = "m",
        thres: float = 0.1,
        min_dist: int = 5,
        fmin: float = 0,
        fmax: float = 40,
        c: int = 0,
        log: bool = False,
        z: int = 0,
    ) -> None:
        Peak = namedtuple("Peak", ["idx", "freq", "amp"])

        def get_peaks(x: npf32, y: npf32) -> List[Peak]:
            idx = indexes(y, thres=thres, min_dist=min_dist)
            peak_amp = [y[i] for i in idx]
            freqs = [x[i] for i in idx]
            return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]

        def plot_spectra(ax: Axes, x: npf32, y: npf32, peaks: List[Peak]) -> None:
            ax.plot(x, y)
            for _, freq, amp in peaks:
                ax.text(
                    freq,
                    amp + 0.03 * max(y),  # type: ignore
                    f"{freq:.2f}",
                    # fontsize=5,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
            ax.set_title(self.name)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        def plot_modes(axes: axType, f: float) -> None:
            if not isinstance(axes, np.ndarray):
                axes = np.array([[axes]])
            for ax in axes.flatten():
                ax.cla()
                ax.set(xticks=[], yticks=[])
            mode = self.get_mode(dset, f)[z]
            extent = [
                0,
                mode.shape[1] * self.dx * 1e9,  # type: ignore
                0,
                mode.shape[0] * self.dy * 1e9,  # type: ignore
            ]
            for c in range(3):
                abs_arr = np.abs(mode[:, :, c])  # type: ignore
                phase_arr = np.angle(mode[:, :, c])  # type: ignore
                axes[0, c].imshow(
                    abs_arr,
                    cmap="inferno",
                    vmin=0,
                    vmax=abs_arr.max(),
                    # norm=mpl.colors.LogNorm(vmin=5e-3),
                    extent=extent,
                    interpolation="None",
                    aspect="equal",
                )
                axes[1, c].imshow(
                    phase_arr,
                    aspect="equal",
                    cmap="hsv",
                    vmin=-np.pi,
                    vmax=np.pi,
                    interpolation="None",
                    extent=extent,
                )
                axes[2, c].imshow(
                    phase_arr,
                    aspect="equal",
                    alpha=abs_arr / abs_arr.max(),
                    cmap="hsv",
                    vmin=-np.pi,
                    vmax=np.pi,
                    interpolation="nearest",
                    extent=extent,
                )

        def get_spectrum():
            x = self.get_dset(f"fft/{dset}/freqs")[(slice(None),)]
            y = self.get_dset(f"fft/{dset}/spec")[(slice(None), c)]
            if log:
                y = np.log(y)
                y -= y.min()
                y /= y.max()
            if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
                raise ValueError("x and y must be numpy arrays")
            x1 = np.abs(x - fmin).argmin()  # type: ignore
            x2 = np.abs(x - fmax).argmin()  # type: ignore
            return x[x1:x2], y[x1:x2]

        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = fig.add_gridspec(1, 2)
        ax_spec = fig.add_subplot(gs[0, 0])
        x, y = get_spectrum()
        peaks = get_peaks(x, y)  # type: ignore
        plot_spectra(ax_spec, x, y, peaks)  # type: ignore
        axes_modes = gs[0, 1].subgridspec(3, 3).subplots()
        vline = ax_spec.axvline(10, ls="--", lw=0.8, c="#ffb86c")
        # plot_modes(axes_modes, peaks[0].freq)

        def onclick(event: Event) -> None:
            if isinstance(event, MouseEvent):
                if event.inaxes == ax_spec:
                    f = 10.0
                    if event.button == 3:  # Right mouse button
                        freqs = np.array([p.freq for p in peaks])
                        f = freqs[(np.abs(freqs - event.xdata)).argmin()]
                    else:
                        xdata = event.xdata
                        if not isinstance(xdata, float):
                            return
                        f = xdata
                    vline.set_data([f, f], [0, 1])
                    plot_modes(axes_modes, f)
                    fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

    def snapshot(
        self,
        dset: str = "m",
        z: int = 0,
        t: int = -1,
        ax: Optional[Axes] = None,
        repeat: int = 1,
        zero: Optional[bool] = None,
    ) -> Axes:
        arr = self.get_dset(dset)[(t, z, slice(None), slice(None), slice(None))]
        if ax is None:
            shape_ratio = arr.shape[1] / arr.shape[0]
            _, ax = plt.subplots(1, 1, figsize=(3 * shape_ratio, 3), dpi=100)
        if zero is not None:
            arr -= self.get_dset(dset)[(zero, z, slice(None), slice(None), slice(None))]  # type: ignore
        arr = np.tile(arr, (repeat, repeat, 1))
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        w = arr[:, :, 2]

        alphas = -np.abs(w) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float32)
        hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2  # type: ignore
        hsl[:, :, 1] = np.sqrt(u**2 + v**2 + w**2)  # type: ignore
        hsl[:, :, 2] = (w + 1) / 2  # type: ignore
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[1] / 20), 1)
        stepy = max(int(u.shape[0] / 20), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * float(self.dx) * 1e9,
            np.arange(0, u.shape[0], stepy) * float(self.dy) * 1e9,
        )
        adset = self.z[dset]
        if not isinstance(adset, zarr.Array):
            raise ValueError("dset must be a zarr array")
        antidots = np.ma.masked_not_equal(adset[0, 0, :, :, 2], 0)
        antidots = np.tile(antidots, (repeat, repeat))
        ax.quiver(
            x,
            y,
            u[::stepy, ::stepx],
            v[::stepy, ::stepx],
            alpha=alphas[::stepy, ::stepx],
            angles="xy",
            scale_units="xy",
            scale=scale,
        )
        ax.imshow(
            rgb,
            interpolation="None",
            origin="lower",
            aspect="equal",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=(
                0,
                rgb.shape[1] * float(self.dx) * 1e9,
                0,
                rgb.shape[0] * float(self.dy) * 1e9,
            ),
        )
        ax.set(title=self.name, xlabel="x (nm)", ylabel="y (nm)")
        if not isinstance(ax, Axes):
            raise ValueError("ax must be None")
        return ax

    def trim_modes(
        self, dset_in_str: str = "m", dset_out_str: str = "m", peak_xcut_min: int = 0
    ) -> None:
        self.rm(f"tmodes/{dset_in_str}")
        dset_in = self.get_dset(f"modes/{dset_in_str}/arr")
        all_peaks = []
        for c in range(3):
            spec = self.get_dset("fft/m/freqs")[(slice(peak_xcut_min, None), c)]
            peaks = np.array([])
            for thres in np.linspace(0.1, 0.001):
                peaks = indexes(spec / spec.max(), thres=thres, min_dist=2)
                if len(peaks) > 25:
                    break
            if len(peaks) == 0:
                print(f"No peaks found for {c=} in {self.name}")
            for p in peaks:
                all_peaks.append(p + peak_xcut_min)
        ap = np.asarray(all_peaks)
        ap = np.unique(ap)
        ap.sort(axis=0)
        s = dset_in.shape
        self.z.create_dataset(
            f"tmodes/{dset_out_str}/freqs",
            data=np.array([self.z[f"modes/{dset_in_str}/freqs"][i] for i in ap]),
            chunks=False,
            dtype=np.float32,
        )
        self.z.create_dataset(
            f"tmodes/{dset_out_str}/arr",
            data=np.array([dset_in[i] for i in ap], np.complex64),
            chunks=(1, *s[1:]),
            dtype=np.complex64,
        )

    def ihist(
        self, dset: str = "m", xdata: str = "B_extz", ydata: str = "mz", z: int = 0
    ) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        fig.subplots_adjust(bottom=0.16, top=0.94, right=0.99, left=0.08)
        x = self.get_dset(f"table/{xdata}")[(slice(None),)]
        y = self.get_dset(f"table/{ydata}")[(slice(None),)]
        ax1.plot(x, y)
        ax1.set_xlabel(xdata)
        ax1.set_ylabel(ydata)
        selector = x.shape[0] // 4
        vline = ax1.axvline(x[selector], c="gray", ls=":")
        hline = ax1.axhline(y[selector], c="gray", ls=":")
        ax1.grid()

        def onclick(e: Event) -> None:
            if isinstance(e, MouseEvent):
                if e.inaxes == ax1:
                    ax2.cla()
                    xdata = e.xdata
                    ydata = e.ydata
                    if not isinstance(xdata, float) or not isinstance(ydata, float):
                        return
                    i = get_closest_point_on_fig(xdata, ydata, x, y, fig)
                    self.snapshot(dset, t=i, ax=ax2, z=z)
                    ax2.set_title("")
                    vline.set_data([x[i], x[i]], [-1, 1])
                    hline.set_data([x.min(), x.max()], [y[i], y[i]])
                    fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)
