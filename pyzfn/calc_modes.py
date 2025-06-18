from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyzfn import Pyzfn


def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    slices: tuple[slice, ...] | slice = (slice(None),) * 5,
    window: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Calculate spatially-resolved FFT modes and store the results in-place.
    This function computes the FFT of a 5-D dataset (time, z, y, x, c) and stores
    the results in a structured format under the `fft` and `modes` namespaces.
    Parameters
    ----------
    dset_in_str : str
        Name of the input dataset to process.
    dset_out_str : str
        Name of the output dataset to create.
    slices : tuple[slice, ...] | slice
        Slices to apply to the input dataset. Defaults to all data. Tip: use np.s_ to create complex slices.
    window : bool
        Whether to apply a Hanning window to the time dimension before FFT.
        Defaults to True.
    overwrite : bool
        Whether to overwrite existing output datasets. Defaults to True.
    Raises
    ------
    ValueError
        If the input dataset does not have the expected shape or lacks the required time attribute.
    FileExistsError
        If the output datasets already exist and `overwrite` is set to False.
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

    if isinstance(slices, slice):
        slices = (slices,)

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
        data=np.max(spec, axis=(1, 2, 3)),
        overwrite=overwrite,
    )
    self.add_ndarray(
        f"fft/{dset_out_str}/sum",
        data=np.sum(spec, axis=(1, 2, 3)),
        overwrite=overwrite,
    )
