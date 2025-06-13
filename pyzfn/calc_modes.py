from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyzfn import Pyzfn


def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    window: bool = True,
    tmin: int | None = None,
    tmax: int | None = None,
    zmin: int | None = None,
    zmax: int | None = None,
    ymin: int | None = None,
    ymax: int | None = None,
    xmin: int | None = None,
    xmax: int | None = None,
    cmin: int | None = None,
    cmax: int | None = None,
    overwrite: bool = True,
) -> None:
    """
    Calculate spatially-resolved FFT modes and store the results in-place.
    """
    # ------------------------------------------------------------------ #
    # 0) Resolve & validate the input dataset
    # ------------------------------------------------------------------ #
    dset_in = self.get_array(dset_in_str)

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
    bounds = np.asarray(
        [tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax, cmin, cmax], dtype=np.int64
    )
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
    if not overwrite and any(p in self for p in targets):
        raise FileExistsError(
            f"Output nodes already exist and overwrite=False: {', '.join(p for p in targets if p in self)}"
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
    ts = np.asarray(dset_in.attrs["t"], np.float64)
    freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9

    # ------------------------------------------------------------------ #
    # 6) Persist results
    # ------------------------------------------------------------------ #
    self.add_ndarray(
        f"modes/{dset_out_str}/t",
        data=ts[tmin:tmax],
        overwrite=overwrite,
    )
    self.add_ndarray(
        f"modes/{dset_out_str}/freqs",
        data=freqs,
        overwrite=overwrite,
    )
    self.add_ndarray(
        f"modes/{dset_out_str}/arr",
        data=out,
        overwrite=overwrite,
        chunks=(1, out.shape[1], out.shape[2], out.shape[3], out.shape[4]),
    )

    spec = np.abs(out)
    self.add_ndarray(
        f"fft/{dset_out_str}/freqs",
        data=freqs,
        overwrite=overwrite,
    )
    self.add_ndarray(
        f"fft/{dset_out_str}/spec",
        data=np.max(spec, axis=(1, 2, 3)).astype(np.float32),
        overwrite=overwrite,
    )
    self.add_ndarray(
        f"fft/{dset_out_str}/sum",
        data=np.sum(spec, axis=(1, 2, 3)).astype(np.float32),
        overwrite=overwrite,
    )
