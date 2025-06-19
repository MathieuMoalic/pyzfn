"""Functions for calculating spatially-resolved FFT modes."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyzfn import Pyzfn

NDIMS = 5


def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    slices: tuple[slice, ...] | slice | None = None,
    *,
    window: bool = True,
) -> None:
    """Calculate spatially-resolved FFT modes and store the results in-place.

    This function computes the FFT of a 5-D dataset (time, z, y, x, c) and stores
    the results in a structured format under the `fft` and `modes` namespaces.

    Parameters
    ----------
    self : Pyzfn
        Instance of the Pyzfn class on which this method operates.
    dset_in_str : str
        Name of the input dataset to process.
    dset_out_str : str
        Name of the output dataset to create.
    slices : tuple[slice, ...] | slice
        Slices to apply to the input dataset. Defaults to all data.
        Tip: use np.s_ to create complex slices.
    window : bool
        Whether to apply a Hanning window to the time dimension before FFT.
        Defaults to True.

    Raises
    ------
    ValueError
        If the input dataset does not have the expected shape or
        lacks the required time attribute.

    Notes
    -----
    This function expects the input dataset to be a 5-D array with dimensions
    (t, z, y, x, c), where:
        - t: time dimension
        - z: spatial dimension (e.g., thickness)
        - y: spatial dimension (e.g., width)
        - x: spatial dimension (e.g., length)
        - c: vector dimension (e.g., magnetization components)
    The output datasets will be structured as follows:
    - `fft/{dset_out_str}/freqs`: Frequencies corresponding to the FFT.
    - `fft/{dset_out_str}/spec`: Maximum spectral amplitude across spatial dimensions.
    - `fft/{dset_out_str}/sum`: Sum of spectral amplitudes across spatial dimensions.
    - `modes/{dset_out_str}/freqs`: Frequencies corresponding to the FFT modes.
    - `modes/{dset_out_str}/arr`: Complex FFT modes array.

    """
    dset_in = self.get_array(dset_in_str)

    if slices is None:
        slices = (slice(None),) * NDIMS
    elif isinstance(slices, slice):
        slices = (slices,)

    if dset_in.ndim != NDIMS:
        msg = f"Expected a 5-D array (t,z,y,x,c); got {dset_in.ndim}-D."
        raise ValueError(msg)

    if "t" not in dset_in.attrs:
        msg = f"Dataset '{dset_in_str}' lacks required time attribute 't'."
        raise ValueError(msg)
    ts = np.asarray(dset_in.attrs["t"], dtype=np.float64)
    if ts.size != dset_in.shape[0]:
        msg = (
            f"len(attrs['t'])={ts.size} does not match time dimension "
            f"{dset_in.shape[0]}"
        )
        raise ValueError(msg)

    arr = np.asarray(dset_in[slices], dtype=np.float32)
    arr -= arr.mean(axis=0, keepdims=True)
    if window:
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]

    out = np.fft.rfft(arr, axis=0).astype(np.complex64)

    time_slice = (
        slices[0] if isinstance(slices, tuple) and len(slices) > 0 else slice(None)
    )
    ts = np.asarray(dset_in.attrs["t"], np.float64)[time_slice]
    freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9

    self.add_ndarray(
        f"modes/{dset_out_str}/freqs",
        data=freqs,
    )
    self.add_ndarray(
        f"modes/{dset_out_str}/arr",
        data=out,
        chunks=(1, out.shape[1], out.shape[2], out.shape[3], out.shape[4]),
    )

    spec = np.abs(out)
    self.add_ndarray(
        f"fft/{dset_out_str}/freqs",
        data=freqs,
    )
    self.add_ndarray(
        f"fft/{dset_out_str}/spec",
        data=np.max(spec, axis=(1, 2, 3)),
    )
    self.add_ndarray(
        f"fft/{dset_out_str}/sum",
        data=np.sum(spec, axis=(1, 2, 3)),
    )
