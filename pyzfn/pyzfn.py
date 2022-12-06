import os
from pathlib import Path
import shutil
import multiprocessing as mp

import numpy as np
import pyfftw
import zarr
from tqdm.notebook import trange

from .utils import get_slices


class Pyzfn:
    """Postprocessing, visualization for amumax's zarr outputs"""

    def __init__(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path Not Found : '{path}'")
        self.z: zarr.Group = zarr.open(path)
        self.path: Path = Path(path).absolute()
        self.name: str = self.path.name.replace(self.path.suffix, "")

    def __getitem__(self, item: str) -> zarr.Array | zarr.Group:
        return self.z[item]

    def __getattr__(self, name: str) -> zarr.Array | zarr.Group | int | float | str:
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
        print(self.z.tree())

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
        slices: tuple[slice, slice, slice, slice, slice] = (
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
        _, _, cy, _, _ = dset_in.chunks
        ts = dset_in.attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        lenf = freqs.shape[0]
        self.z.create_dataset(f"fft/{dset_out_str}/freqs", data=freqs, chunks=False)
        dset_fft = self.z.create_dataset(
            f"fft/{dset_out_str}/max", shape=(lenf, 3), chunks=False
        )
        self.z.create_dataset(f"modes/{dset_out_str}/freqs", data=freqs, chunks=False)
        dset_modes = self.z.create_dataset(
            f"modes/{dset_out_str}/arr",
            shape=(lenf, sz, sy, sx, sc),
            dtype=np.complex64,
            chunks=(1, None, None, None, None),
        )
        x0 = pyfftw.empty_aligned((st, sz, cy, sx, sc), dtype=np.complex64)
        fft = pyfftw.builders.fft(
            x0,
            axis=0,
            threads=mp.cpu_count() // 2,
            planner_effort="FFTW_ESTIMATE",
            avoid_copy=True,
        )
        fft_out = []
        zsls, ysls, xsls, csls = get_slices(dset_in.shape, dset_in.chunks, slices)
        for zsl in zsls:
            for ysl in ysls:
                for xsl in xsls:
                    for csl in csls:
                        x0[:] = dset_in[slices[0], zsl, ysl, xsl, csl]
                        x0 -= np.average(x0, axis=0)[None, ...]
                        x0 *= np.hanning(x0.shape[0])[:, None, None, None, None]
                        x0 = fft()[:lenf]
                        dset_modes[slices[0], zsl, ysl, xsl, csl] = x0
                        x0 = np.abs(x0)
                        x0 = np.max(x0, axis=(1, 2, 3))
                        fft_out.append([x0])
        dset_fft[:] = np.max(fft_out, axis=1)
