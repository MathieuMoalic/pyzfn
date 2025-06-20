"""Functions for calculating spatially-resolved FFT modes with memory checks.

This module provides utilities for estimating memory usage, performing FFTs on
multi-dimensional datasets, and storing the results, with safety checks to
avoid exceeding available system RAM.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import psutil

from .utils import format_bytes

if TYPE_CHECKING:  # pragma: no cover
    from zarr import Array

    from pyzfn import Pyzfn

NDIMS = 5
LOGGER = logging.getLogger(__name__)
if LOGGER.level == logging.NOTSET:  # only set a default the first time
    logging.basicConfig(level=logging.INFO)  # change to DEBUG for more detail


def _normalise_slices(
    user_slices: slice | tuple[slice, ...] | None,
) -> tuple[slice, slice, slice, slice, slice]:
    if user_slices is None:
        slices = (slice(None), slice(None), slice(None), slice(None), slice(None))
    elif isinstance(user_slices, slice):
        slices = (user_slices, slice(None), slice(None), slice(None), slice(None))
    elif isinstance(user_slices, tuple):
        if len(user_slices) > NDIMS:
            msg = (
                f"Too many slices provided: expected at most {NDIMS}, "
                f"got {len(user_slices)}."
            )
            raise ValueError(msg)
        t, z, y, x, c = 0, 1, 2, 3, 4
        slices = (
            user_slices[0] if len(user_slices) > t else slice(None),
            user_slices[1] if len(user_slices) > z else slice(None),
            user_slices[2] if len(user_slices) > y else slice(None),
            user_slices[3] if len(user_slices) > x else slice(None),
            user_slices[4] if len(user_slices) > c else slice(None),
        )
    return slices  # type: ignore[return-value]


def _validate_dataset(dset: "Array", dset_name: str) -> np.ndarray:
    if dset.ndim != NDIMS:
        msg = f"Expected a 5-D array (t,z,y,x,c); got {dset.ndim}-D."
        raise ValueError(msg)

    if "t" not in dset.attrs:
        msg = f"Dataset '{dset_name}' lacks required time attribute 't'."
        raise ValueError(msg)

    ts = np.asarray(dset.attrs["t"], dtype=np.float64)
    if ts.size != dset.shape[0]:
        msg = f"len(attrs['t'])={ts.size} does not match time dimension {dset.shape[0]}"
        raise ValueError(msg)
    return ts


def _input_shape(
    dset: "Array",
    slices: tuple[slice, slice, slice, slice, slice],
) -> tuple[int, int, int, int, int]:
    shape = [0, 0, 0, 0, 0]
    for i, s in enumerate(slices):
        start: int = s.start if s.start is not None else 0
        stop: int = s.stop if s.stop is not None else dset.shape[i]
        shape[i] = stop - start
    return tuple(shape)  # type: ignore[return-value]


def estimate_peak_ram(in_shape: tuple[int, ...]) -> dict[str, int]:
    """Estimate the peak RAM usage for FFT processing of a dataset.

    Parameters
    ----------
    in_shape : tuple of int
        Shape of the input array (typically (t, z, y, x, c)).

    Returns
    -------
    dict of str to int
        Dictionary with estimated memory usage in bytes for the input array ('arr'),
        FFT output ('out'), and spectrum ('spec').

    """
    arr_nel = np.prod(in_shape, dtype=np.int64)
    arr_bytes = int(arr_nel * 4)  # float32

    fft_len = in_shape[0] // 2 + 1
    out_shape = (fft_len, *in_shape[1:])
    out_nel = np.prod(out_shape, dtype=np.int64)
    out_bytes = int(out_nel * 8)  # complex64

    spec_bytes = int(out_nel * 4)  # float32

    return {"arr": arr_bytes, "out": out_bytes, "spec": spec_bytes}


def check_memory(
    est: dict[str, int],
    ratio: float,
    logger: logging.Logger = LOGGER,
) -> None:
    """Check if the estimated peak RAM usage exceeds a safe fraction of available RAM.

    Parameters
    ----------
    est : dict of str to int
        Dictionary with estimated memory usage in bytes for different arrays.
    ratio : float
        Fraction of available system RAM allowed for use (e.g., 0.8 for 80%).
    logger : logging.Logger, optional
        Logger to use for informational messages.

    Raises
    ------
    MemoryError
        If the estimated peak RAM usage exceeds the allowed fraction of available RAM.

    """
    peak = sum(est.values())
    ram_free = psutil.virtual_memory().available
    safe_limit = ram_free * ratio

    logger.info(
        "Estimated peak RAM: %s  (arr=%s, out=%s, spec=%s)",
        format_bytes(peak),
        format_bytes(est["arr"]),
        format_bytes(est["out"]),
        format_bytes(est["spec"]),
    )
    logger.info(
        "Free system RAM: %s  |  safety ratio %.0f%% → allowed: %s",
        format_bytes(ram_free),
        ratio * 100,
        format_bytes(int(safe_limit)),
    )

    if peak > safe_limit:
        msg = (
            "FFT aborted: needs "
            f"{format_bytes(peak)}, but only "
            f"{format_bytes(int(safe_limit))} is allowed under the "
            f"{ratio:.0%} safety margin. "
            "Reduce the dataset size or use a machine with more RAM."
        )
        raise MemoryError(msg)


def _prepare_data(arr: np.ndarray, *, window: bool) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    arr -= arr.mean(axis=0, keepdims=True)
    if window:
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]
    return arr


def _store_results(
    obj: "Pyzfn",
    freqs: np.ndarray,
    out: np.ndarray,
    dset_out_str: str,
) -> None:
    """Write FFT modes and spectra back to *obj*."""
    obj.add_ndarray(f"modes/{dset_out_str}/freqs", data=freqs)
    obj.add_ndarray(
        f"modes/{dset_out_str}/arr",
        data=out,
        chunks=(1, *out.shape[1:]),
    )

    spec = np.abs(out)
    obj.add_ndarray(f"fft/{dset_out_str}/freqs", data=freqs)
    obj.add_ndarray(f"fft/{dset_out_str}/spec", data=np.max(spec, axis=(1, 2, 3)))
    obj.add_ndarray(f"fft/{dset_out_str}/sum", data=np.sum(spec, axis=(1, 2, 3)))


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#
def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    slices: tuple[slice, ...] | slice | None = None,
    *,
    window: bool = True,
) -> None:
    """Calculate spatially-resolved FFT modes with a pre-flight RAM check.

    Parameters
    ----------
    self : Pyzfn
        Instance of the :class:`~pyzfn.Pyzfn` class on which this method operates.
    dset_in_str : str, default ``"m"``
        Name of the input dataset to process.
    dset_out_str : str, default ``"m"``
        Name under which the output datasets will be stored.
    slices : slice or tuple of slices, optional
        Time/space slices to apply to the input dataset.  Use :pydata:`np.s_` for
        convenient construction.  If *None*, the full dataset is used.
    window : bool, default ``True``
        Apply a Hanning window along the time axis before the FFT.
    max_ram_usage_ratio : float, default ``0.8``
        Abort if the estimated peak memory would exceed this fraction of the
        currently *available* system RAM.

    Notes
    -----
    The input dataset must have shape ``(t, z, y, x, c)`` and an attribute
    ``'t'`` with equally-spaced timestamps (in ns).  The following arrays are
    written back into the HDF5 hierarchy::

        /fft/{dset_out_str}/freqs   - 1-D float64
        /fft/{dset_out_str}/spec    - 2-D float32 (freq, c)
        /fft/{dset_out_str}/sum     - 2-D float32 (freq, c)
        /modes/{dset_out_str}/freqs - same as above
        /modes/{dset_out_str}/arr   - 5-D complex64 (freq, z, y, x, c)

    """
    dset_in = self.get_array(dset_in_str)
    ts_full = _validate_dataset(dset_in, dset_in_str)
    slices = _normalise_slices(slices)

    in_shape = _input_shape(dset_in, slices)
    est = estimate_peak_ram(in_shape)
    max_ram_usage_ratio = 0.8
    check_memory(est, max_ram_usage_ratio)

    LOGGER.info(
        "inner_calc_modes input shape %s → sliced shape %s (max %.0f %% RAM rule)",
        dset_in.shape,
        in_shape,
        max_ram_usage_ratio * 100,
    )

    arr = _prepare_data(np.asarray(dset_in[slices]), window=window)

    out = np.fft.rfft(arr, axis=0).astype(np.complex64)

    time_slice = slices[0] if slices else slice(None)
    ts = ts_full[time_slice]
    freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9

    _store_results(self, freqs, out, dset_out_str)

    LOGGER.info("FFT complete and data written to 'fft' and 'modes' namespaces.")
