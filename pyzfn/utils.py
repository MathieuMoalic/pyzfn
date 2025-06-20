"""Utility functions for color manipulation, peak finding, and formatting byte sizes.

This module provides:
- Color conversion utilities (HSL/RGB)
- Custom colormap creation
- Peak detection in 1D signals
- Byte size formatting
"""

import colorsys
from typing import Literal, NamedTuple, TypeVar

import matplotlib.colors as mcolors
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)
NPArray = NDArray[np.generic]


def make_cmap(
    min_color: tuple[float, float, float, float],
    max_color: tuple[float, float, float, float],
    mid_color: tuple[float, float, float, float] | None = None,
    *,
    transparent_zero: bool = False,
) -> mcolors.ListedColormap:
    """Create a matplotlib ListedColormap with interpolating.

    Parameters
    ----------
    min_color : tuple[float, float, float, float]
        RGBA color tuple for the minimum value.
    max_color : tuple[float, float, float, float]
        RGBA color tuple for the maximum value.
    mid_color : tuple[float, float, float, float] or None, optional
        RGBA color tuple for the midpoint value. If None, a linear gradient is used.
    transparent_zero : bool, optional
        If True, sets the alpha of the midpoint to 0 for transparency.

    Returns
    -------
    matplotlib.colors.ListedColormap
        The resulting colormap.

    """
    cmap: NDArray[np.float32] = np.ones((256, 4), dtype=np.float32)
    for i in range(4):
        if mid_color is None:
            cmap[:, i] = np.linspace(min_color[i], max_color[i], 256) / 256
        else:
            cmap[:128, i] = np.linspace(min_color[i], mid_color[i], 128) / 256
            cmap[128:, i] = np.linspace(mid_color[i], max_color[i], 128) / 256
    if transparent_zero:
        cmap[128, 3] = 0.0
    return mcolors.ListedColormap(cmap.tolist())  # Avoid passing ndarray directly


def hsl2rgb(hsl: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an array of HSL values to RGB.

    Parameters
    ----------
    hsl : NDArray[np.float64]
        An array of shape (..., 3) containing HSL values in the range [0, 1].

    Returns
    -------
    NDArray[np.float64]
        An array of the same shape as input, containing RGB values in the range [0, 1].

    """
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]  # noqa: E741
    rgb = np.zeros_like(hsl)
    for i, n in enumerate([0, 8, 4]):
        k = (n + h / 30) % 12
        a = s * np.minimum(l, 1 - l)
        k = np.minimum(k - 3, 9 - k)
        k = np.clip(k, -1, 1)
        rgb[..., i] = l - a * k
    return np.clip(rgb, 0, 1)


def rgb2hsl(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an array of RGB values to HSL.

    Parameters
    ----------
    rgb : NDArray[np.float64]
        An array of shape (..., 3) containing RGB values in the range [0, 1].

    Returns
    -------
    NDArray[np.float64]
        An array of the same shape as input, containing HSL values in the range [0, 1].

    """
    hsl = np.ones_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            r, g, b = rgb[i, j]
            h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
            hsl[i, j, 0] = h
            hsl[i, j, 1] = s
            hsl[i, j, 2] = l
    return hsl


class Peak(NamedTuple):
    """A named tuple representing a peak in a signal.

    Attributes
    ----------
    frequency : float
        The frequency at which the peak occurs.
    amplitude : float
        The amplitude of the peak.
    idx : int
        The index of the peak in the signal array.

    """

    frequency: float
    amplitude: float
    idx: int


Numeric = np.integer | np.floating


# ───────────────────────────── helpers ──────────────────────────────
def _validate_1d_same_shape(freq: NDArray, sig: NDArray) -> None:
    if sig.ndim != 1 or freq.ndim != 1:
        msg = "signal and frequencies must be 1-D"
        raise ValueError(msg)
    if sig.shape != freq.shape:
        msg = "signal and frequencies must have the same shape"
        raise ValueError(msg)


def _numeric_threshold(sig: NDArray, threshold: float, n_peaks: int | None) -> float:
    if n_peaks is not None:  # 'grab everything' mode
        return sig.min() - np.finfo(sig.dtype).eps
    return threshold * (sig.max() - sig.min()) + sig.min()


def _handle_plateaus(dy: np.ndarray) -> np.ndarray:
    zeros = np.where(dy == 0)[0]
    if not len(zeros):
        return dy
    zeros_diff = np.diff(zeros)
    zeros_diff_not_one = np.add(np.where(zeros_diff != 1), 1)[0]
    zero_plateaus = np.split(zeros, zeros_diff_not_one)

    if zero_plateaus and zero_plateaus[0][0] == 0:
        dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
        zero_plateaus.pop(0)
    if zero_plateaus and zero_plateaus[-1][-1] == len(dy) - 1:
        dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
        zero_plateaus.pop(-1)
    for plateau in zero_plateaus:
        median = np.median(plateau)
        dy[plateau[plateau < median]] = dy[plateau[0] - 1]
        dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]
    return dy


def _detect_raw_peaks(sig: NDArray, thr: float) -> NDArray[np.int64]:
    dy = _handle_plateaus(np.diff(sig))
    return np.where(
        (np.hstack([dy, 0.0]) < 0)  # falling edge
        & (np.hstack([0.0, dy]) > 0)  # rising edge
        & (sig > thr),  # above threshold
    )[0]


def _apply_min_dist(
    peaks: NDArray[np.int64],
    sig: NDArray,
    min_dist: int,
) -> NDArray[np.int64]:
    if peaks.size <= 1 or min_dist <= 1:
        return peaks
    order = peaks[np.argsort(sig[peaks])][::-1]  # high → low
    keep = np.ones(sig.size, dtype=bool)
    keep[peaks] = False
    for p in order:
        if not keep[p]:
            sl = slice(max(0, p - min_dist), p + min_dist + 1)
            keep[sl] = True
            keep[p] = False
    return np.where(~keep)[0]


def _prune_to_top_n(
    peaks: NDArray[np.int64],
    sig: NDArray,
    n: int | None,
) -> NDArray[np.int64]:
    if n is None or peaks.size <= n:
        return peaks
    return peaks[np.argsort(sig[peaks])][-n:]  # largest amplitudes


# ──────────────────────────── public API ────────────────────────────
def find_peaks(  # noqa: PLR0913
    frequencies: NDArray[Numeric],
    signal: NDArray[Numeric],
    *,
    threshold: float = 0.3,
    min_dist: int = 1,
    n_peaks: int | None = None,
    sort_by: Literal["frequency", "amplitude"] = "frequency",
) -> list[Peak]:
    """Detect peaks in a 1-D signal.

    Parameters
    ----------
    frequencies, signal
        1-D arrays of the same length.
    threshold
        Relative (0-1) or absolute value. Ignored when *n_peaks* is given.
    min_dist
        Minimum index distance between successive peaks.
    n_peaks
        If set, keep lowering the threshold until **≥ n_peaks** candidates exist,
        then return the *n_peaks* highest-amplitude peaks (after *min_dist* pruning).
    sort_by
        Order result by ascending 'frequency' (default) or descending 'amplitude'.

    Returns
    -------
    list[Peak]
        Each peak holds `(frequency, amplitude, idx)`.

    Raises
    ------
    ValueError
        If the threshold is not in the range [0, 1] or a positive absolute value,
        or if the input arrays are not 1-D or do not have the same shape.

    """
    _validate_1d_same_shape(frequencies, signal)
    if not frequencies.size or not signal.size:
        return []
    if threshold < 0 or threshold > 1:
        msg = "threshold must be in the range [0, 1] or a positive absolute value"
        raise ValueError(msg)
    thr = _numeric_threshold(signal, threshold, n_peaks)
    peaks = _detect_raw_peaks(signal, thr)
    peaks = _apply_min_dist(peaks, signal, min_dist)
    peaks = _prune_to_top_n(peaks, signal, n_peaks)

    peaks_list = [Peak(float(frequencies[i]), float(signal[i]), int(i)) for i in peaks]
    if sort_by == "amplitude":
        peaks_list.sort(key=lambda p: p.amplitude, reverse=True)
    else:  # 'frequency'
        peaks_list.sort(key=lambda p: p.frequency)
    return peaks_list


BYTES_PER_KIB = 1024


def format_bytes(byte_size: int) -> str:
    """Format a byte size into a human-readable string with binary prefixes.

    Parameters
    ----------
    byte_size : int
        The size in bytes to format.

    Returns
    -------
    str
        The formatted string representing the byte size with appropriate units.

    """
    if byte_size < BYTES_PER_KIB:
        return f"{byte_size} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    size = float(byte_size)
    unit_index = 0
    while size >= BYTES_PER_KIB and unit_index < len(units) - 1:
        size /= BYTES_PER_KIB
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"
