import os
from pathlib import Path
import shutil
import multiprocessing as mp
from collections import namedtuple
from typing import Optional, Union, Tuple, List
import peakutils

import numpy as np
import pyfftw
import zarr
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
from nptyping import NDArray, Shape, Float32

from .utils import get_slices, hsl2rgb

np1d = NDArray[Shape["*"], Float32]
np2d = NDArray[Shape["*,*"], Float32]
np3d = NDArray[Shape["*,*,*"], Float32]
np4d = NDArray[Shape["*,*,*,*"], Float32]
np5d = NDArray[Shape["*,*,*,*,*"], Float32]


class Pyzfn:
    """Postprocessing, visualization for amumax's zarr outputs"""

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path Not Found : '{path}'")
        self.z: zarr.Group = zarr.open(path)
        self.path: Path = Path(path).absolute()
        self.name: str = self.path.name.replace(self.path.suffix, "")

    def __getitem__(self, item: str) -> Union[zarr.Array, zarr.Group]:
        return self.z[item]

    def __getattr__(self, name: str) -> Union[zarr.Array, zarr.Group, int, float, str]:
        if name in dir(self):
            return getattr(self, name)
        if name in dir(self.z):
            return getattr(self.z, name)
        if name in self.z.attrs:
            return self.z.attrs[name]
        raise KeyError(name)

    def __repr__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __str__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __name__(self) -> str:
        return self.name

    @property
    def pp(self) -> zarr.util.TreeViewer:
        """Pretty print the tree"""
        return self.z.tree(expand=True)

    @property
    def p(self) -> None:
        """Print the tree"""
        print(self.name, self.z.tree())

    def rm(self, dset: str) -> None:
        """
        Remove a group or dataset
        :param dset: str:

        """
        shutil.rmtree(f"{self.path}/{dset}", ignore_errors=True)

    def mkdir(self, name: str) -> None:
        """
        Create nested directories
        :param name: str:

        """
        os.makedirs(f"{self.path}/{name}", exist_ok=True)

    def calc_disp(self, dset_in_str: str = "m", dset_out_str: str = "m") -> None:
        """
        Calculate the dispersion of `/dset_in_str` and saves it in `/disp/dset_out_str`
        :param dset_in_str: str:  (Default value = "m")
        :param dset_out_str: str:  (Default value = "m")

        """
        dset_in: zarr.Array = self[dset_in_str]
        dset_out: str = f"disp/{dset_out_str}"
        self.rm(dset_out)
        s = dset_in.shape
        arr_out = np.zeros((s[0] // 2 - 1, s[3] - 1, s[4]), dtype=np.float32)
        hann2d = np.sqrt(np.outer(np.hanning(s[0] - 1), np.hanning(s[3] - 1)))[
            :, :, None
        ]
        arr = pyfftw.empty_aligned((s[0] - 1, s[3] - 1, s[4]), dtype=np.complex64)
        fft = pyfftw.builders.fftn(
            arr,
            axes=(1, 0),
            threads=mp.cpu_count() // 2,
            planner_effort="FFTW_ESTIMATE",
            avoid_copy=True,
        )
        for iy in trange(
            dset_in.shape[2] // dset_in.chunks[2],
            leave=False,
            desc=f"Calculating the dispersion for `{self.name}`",
        ):
            arr[:] = dset_in[1:, 0, iy, 1:, :]
            arr *= hann2d
            out = fft()
            out -= np.average(out, axis=(0, 1))[None, None, :]
            out = out[: out.shape[0] // 2]
            out = np.fft.fftshift(out, axes=(1, 2))
            arr_out[:] += np.abs(out)
        self.z.create_dataset(f"{dset_out}/arr", data=arr_out)
        ts = dset_in.attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        self.z.create_dataset(f"{dset_out}/freqs", data=freqs, chunks=None)
        kvecs = (
            np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.z.attrs["dx"]))
            * 2
            * np.pi
        )
        self.z.create_dataset(f"{dset_out}/kvecs", data=kvecs, chunks=None)

    def calc_modes(
        self,
        dset_in_str: str = "m",
        dset_out_str: str = "m",
        slices: Tuple[slice, slice, slice, slice, slice] = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(None),
        ),
    ) -> None:
        """
        Calculate the spin wave spectra and also the mode profiles of `/dset_in_str`
        and saves it in `/disp/dset_out_str`
        :param dset_in_str: str:  (Default value = "m")
        :param dset_out_str: str:  (Default value = "m")

        """
        dset_in: zarr.Array = self[dset_in_str]
        self.rm(f"modes/{dset_out_str}")
        self.rm(f"fft/{dset_out_str}")
        st, sz, sy, sx, sc = dset_in.shape
        _, cz, cy, cx, cc = dset_in.chunks
        ts = dset_in.attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        lenf = freqs.shape[0]
        self.z.create_dataset(f"fft/{dset_out_str}/freqs", data=freqs, chunks=False)
        dset_fft = self.z.create_dataset(
            f"fft/{dset_out_str}/spec", shape=(lenf, 3), chunks=False
        )
        self.z.create_dataset(f"modes/{dset_out_str}/freqs", data=freqs, chunks=False)
        dset_modes = self.z.create_dataset(
            f"modes/{dset_out_str}/arr",
            shape=(lenf, sz, sy, sx, sc),
            dtype=np.complex64,
            chunks=(1, None, None, None, None),
        )
        x0 = pyfftw.empty_aligned((st, cz, cy, cx, cc), dtype=np.complex64)
        fft = pyfftw.builders.fft(
            x0,
            axis=0,
            threads=mp.cpu_count() // 2,
            planner_effort="FFTW_ESTIMATE",
            avoid_copy=True,
        )
        fft_out = []
        zsls, ysls, xsls, csls = get_slices(dset_in.shape, dset_in.chunks, slices)
        chunk_nb = len(zsls) * len(ysls) * len(xsls) * len(csls)
        chunk_idx = 0
        with tqdm(total=chunk_nb, desc="Calculating SW modes", leave=False) as pbar:
            for zsl in zsls:
                for ysl in ysls:
                    for xsl in xsls:
                        for csl in csls:
                            x0[:] = dset_in[slices[0], zsl, ysl, xsl, csl]
                            x0 -= np.average(x0, axis=0)[None, ...]
                            x0 *= np.hanning(x0.shape[0])[:, None, None, None, None]
                            x1 = fft()[:lenf]
                            dset_modes[slices[0], zsl, ysl, xsl, csl] = x1
                            x1 = np.abs(x1)
                            x1 = np.max(x1, axis=(1, 2, 3))
                            fft_out.append(x1)
                            chunk_idx += 1
                            pbar.update(chunk_idx)
        dset_fft[:] = np.max(np.array(fft_out), axis=0)

    def get_mode(self, dset: str, f: float, c: Union[int, None] = None) -> np4d:
        fi = int((np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin())
        arr: np4d = self[f"modes/{dset}/arr"][fi]
        if c is None:
            return arr
        else:
            return arr[..., c]

    def ispec(
        self,
        dset: str = "m",
        thres: float = 0.1,
        min_dist: int = 5,
        xmin: float = 0,
        xmax: float = 40,
        c: int = 0,
    ) -> None:
        Peak = namedtuple("Peak", ["idx", "freq", "amp"])

        def get_peaks(x: np1d, y: np1d) -> List[Peak]:
            idx = peakutils.indexes(y, thres=thres, min_dist=min_dist)
            peak_amp = [y[i] for i in idx]
            freqs = [x[i] for i in idx]
            return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]

        def plot_spectra(ax: plt.Axes, x: np1d, y: np1d, peaks: List[Peak]) -> None:
            ax.plot(x, y)
            for _, freq, amp in peaks:
                ax.text(
                    freq,
                    amp + 0.03 * max(y),
                    f"{freq:.2f}",
                    # fontsize=5,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
            ax.set_title(self.name)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        def plot_modes(axes: plt.Axes, f: float) -> None:
            for ax in axes.flatten():
                ax.cla()
                ax.set(xticks=[], yticks=[])
            mode = self.get_mode("m", f)[0]
            extent = [
                0,
                mode.shape[1] * self.dx * 1e9,
                0,
                mode.shape[0] * self.dy * 1e9,
            ]
            for c in range(3):
                abs_arr = np.abs(mode[:, :, c])
                phase_arr = np.angle(mode[:, :, c])
                axes[0, c].imshow(
                    abs_arr,
                    cmap="inferno",
                    # vmin=0,
                    # vmax=mode_list_max,
                    norm=mpl.colors.LogNorm(vmin=5e-3),
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

        def get_spectrum() -> Tuple[np1d, np1d]:
            x = self[f"fft/{dset}/freqs"][:]
            y = self[f"fft/{dset}/spec"][:, c]
            x1 = np.abs(x - xmin).argmin()
            x2 = np.abs(x - xmax).argmin()
            return x[x1:x2], y[x1:x2]

        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = fig.add_gridspec(1, 2)
        ax_spec = fig.add_subplot(gs[0, 0])
        x, y = get_spectrum()
        peaks = get_peaks(x, y)
        plot_spectra(ax_spec, x, y, peaks)
        axes_modes = gs[0, 1].subgridspec(3, 3).subplots()
        vline = ax_spec.axvline(10, ls="--", lw=0.8, c="#ffb86c")
        # plot_modes(axes_modes, peaks[0].freq)

        def onclick(event: mpl.backend_bases.Event) -> None:
            if event.inaxes == ax_spec:
                f = 10
                if event.button.name == "RIGHT":
                    freqs = [p.freq for p in peaks]
                    f = freqs[(np.abs(freqs - event.xdata)).argmin()]
                else:
                    f = event.xdata
                vline.set_data([f, f], [0, 1])
                plot_modes(axes_modes, f)
                fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)

    def snapshot(
        self,
        dset: str = "m",
        z: int = 0,
        t: int = -1,
        ax: Optional[plt.Axes] = None,
        repeat: int = 1,
        zero: Optional[bool] = None,
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        else:
            fig = ax.figure
        arr = self[dset][t, z, :, :, :]
        if zero is not None:
            arr -= self[dset][zero, z, :, :, :]
        arr = np.tile(arr, (repeat, repeat, 1))
        arr = np.ma.masked_equal(arr, 0)
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        z = arr[:, :, 2]

        alphas = -np.abs(z) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float32)
        hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2  # normalization
        hsl[:, :, 1] = np.sqrt(u**2 + v**2 + z**2)
        hsl[:, :, 2] = (z + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[1] / 60), 1)
        stepy = max(int(u.shape[0] / 60), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * float(self.dx) * 1e9,
            np.arange(0, u.shape[0], stepy) * float(self.dy) * 1e9,
        )
        antidots = np.ma.masked_not_equal(self[dset][0, 0, :, :, 2], 0)
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
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=[
                0,
                rgb.shape[1] * float(self.dx) * 1e9,
                0,
                rgb.shape[0] * float(self.dy) * 1e9,
            ],
        )
        ax.imshow(
            antidots,
            interpolation="None",
            origin="lower",
            cmap="Set1_r",
            extent=[
                0,
                arr.shape[1] * float(self.dx) * 1e9,
                0,
                arr.shape[0] * float(self.dy) * 1e9,
            ],
        )
        ax.set(title=self.name, xlabel="x (nm)", ylabel="y (nm)")
        fig.tight_layout()
        return ax
