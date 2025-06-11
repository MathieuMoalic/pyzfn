from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pyzfn import Pyzfn
import zarr

from .utils import (
    check_memory,
)


def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    window: bool = True,
    tmin: Optional[int] = None,
    tmax: Optional[int] = None,
    zmin: Optional[int] = None,
    zmax: Optional[int] = None,
    ymin: Optional[int] = None,
    ymax: Optional[int] = None,
    xmin: Optional[int] = None,
    xmax: Optional[int] = None,
    cmin: Optional[int] = None,
    cmax: Optional[int] = None,
    skip_memory_check: bool = False,
    overwrite: bool = True,
) -> None:
    """
    Calculate spatially-resolved FFT modes and store the results in-place.
    """
    # ------------------------------------------------------------------ #
    # 0) Resolve & validate the input dataset
    # ------------------------------------------------------------------ #
    if dset_in_str not in self.z:
        raise KeyError(f"Dataset '{dset_in_str}' not found in store.")
    dset_in = self.z[dset_in_str]
    if not isinstance(dset_in, zarr.Array):
        raise ValueError(f"'{dset_in_str}' must be a dataset, not a group.")
    if dset_in.ndim != 5:
        raise ValueError(f"Expected a 5-D array (t,z,y,x,c); got {dset_in.ndim}-D.")

    if "t" not in dset_in.attrs:
        raise ValueError(f"Dataset '{dset_in_str}' lacks required time attribute 't'.")
    ts = np.asarray(dset_in.attrs["t"], dtype=np.float64)
    if ts.size != dset_in.shape[0]:
        raise ValueError(
            f"len(attrs['t'])={ts.size} does not match time dimension {dset_in.shape[0]}"
        )

    full_shape: tuple[int, ...] = dset_in.shape  # (t, z, y, x, c)
    print(f"Full shape: {full_shape}")
    # Replace *None* with bounds
    tmax = full_shape[0] if tmax is None else tmax
    zmax = full_shape[1] if zmax is None else zmax
    ymax = full_shape[2] if ymax is None else ymax
    xmax = full_shape[3] if xmax is None else xmax
    cmax = full_shape[4] if cmax is None else cmax
    tmin = 0 if tmin is None else tmin
    zmin = 0 if zmin is None else zmin
    ymin = 0 if ymin is None else ymin
    xmin = 0 if xmin is None else xmin
    cmin = 0 if cmin is None else cmin

    # ------------------------------------------------------------------ #
    # 1) Normalise and check slice bounds
    # ------------------------------------------------------------------ #
    bounds = np.asarray([tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax, cmin, cmax])
    if np.any(bounds < 0):
        raise ValueError("Slice indices must be non-negative.")

    # Range checks & zero-length checks
    for low, high, name, dim in [
        (tmin, tmax, "t", 0),
        (zmin, zmax, "z", 1),
        (ymin, ymax, "y", 2),
        (xmin, xmax, "x", 3),
        (cmin, cmax, "c", 4),
    ]:
        if low >= high:
            raise ValueError(f"Slice '{name}min' must be smaller than '{name}max'.")
        if high > full_shape[dim]:
            raise IndexError(
                f"'{name}max'={high} exceeds dataset shape ({full_shape[dim]})"
            )

    # ------------------------------------------------------------------ #
    # 2) Abort early if datasets exist and overwrite=False
    # ------------------------------------------------------------------ #
    targets = [
        f"fft/{dset_out_str}/freqs",
        f"fft/{dset_out_str}/spec",
        f"fft/{dset_out_str}/sum",
        f"modes/{dset_out_str}/freqs",
        f"modes/{dset_out_str}/arr",
    ]
    if not overwrite and any(p in self.z for p in targets):
        raise FileExistsError(
            f"Output nodes already exist and overwrite=False: {', '.join(p for p in targets if p in self.z)}"
        )

    # ------------------------------------------------------------------ #
    # 3) Optional memory warning
    # ------------------------------------------------------------------ #
    slice_spec = [
        slice(tmin, tmax),
        slice(zmin, zmax),
        slice(ymin, ymax),
        slice(xmin, xmax),
        slice(cmin, cmax),
    ]
    warn_msg = check_memory([tuple(slice_spec)], full_shape, force=skip_memory_check)
    if warn_msg:
        print(warn_msg)

    # ------------------------------------------------------------------ #
    # 4) FFT along the *time* axis
    # ------------------------------------------------------------------ #
    if (tmax - tmin) < 2:
        raise ValueError("Need at least two time frames for rFFT.")

    arr = np.asarray(dset_in[tuple(slice_spec)], dtype=np.float32)
    arr -= arr.mean(axis=0, keepdims=True)
    if window:
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]

    out = np.fft.rfft(arr, axis=0).astype(np.complex64)

    # ------------------------------------------------------------------ #
    # 5) Spectra & frequency axis
    # ------------------------------------------------------------------ #
    ts = np.asarray(dset_in.attrs["t"])
    freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9

    # ------------------------------------------------------------------ #
    # 6) Persist results
    # ------------------------------------------------------------------ #
    self.z.create_dataset(
        f"modes/{dset_out_str}/freqs",
        data=freqs,
        dtype="float32",
        overwrite=overwrite,
        chunks=False,
    )
    self.z.create_dataset(
        f"modes/{dset_out_str}/arr",
        data=out,
        dtype="complex64",
        overwrite=overwrite,
        chunks=(1, None, None, None, None),
    )

    spec = np.abs(out)
    self.z.create_dataset(
        f"fft/{dset_out_str}/freqs",
        data=freqs,
        dtype="float32",
        overwrite=overwrite,
        chunks=False,
    )
    self.z.create_dataset(
        f"fft/{dset_out_str}/spec",
        data=np.max(spec, axis=(1, 2, 3)).astype(np.float32),
        overwrite=overwrite,
        chunks=False,
    )
    self.z.create_dataset(
        f"fft/{dset_out_str}/sum",
        data=np.sum(spec, axis=(1, 2, 3)).astype(np.float32),
        overwrite=overwrite,
        chunks=False,
    )
