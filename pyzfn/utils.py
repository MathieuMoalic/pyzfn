"""Utility functions for color manipulation, peak finding, and formatting byte sizes.

This module provides:
- Color conversion utilities (HSL/RGB)
- Custom colormap creation
- Peak detection in 1D signals
- Byte size formatting
"""

import colorsys
from typing import NamedTuple, TypeVar

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


def find_peaks(
    frequencies: NDArray[Numeric],
    signal: NDArray[Numeric],
    thres: float = 0.3,
    min_dist: int = 1,
    *,
    thres_abs: bool = False,
) -> list[Peak]:
    """Find peaks in a 1D signal array.

    Parameters
    ----------
    frequencies : NDArray[Numeric]
        Array of frequency values corresponding to the signal.
    signal : NDArray[Numeric]
        1D array of signal values.
    thres : float, optional
        Threshold for peak detection. If thres_abs is False, interpreted
        as a fraction of the signal range.
    min_dist : int, optional
        Minimum distance (in samples) between peaks.
    thres_abs : bool, optional
        If True, threshold is interpreted as an absolute value.

    Returns
    -------
    list[Peak]
        List of detected peaks as Peak namedtuples.

    Raises
    ------
    ValueError
        If input arrays do not have the same shape or are not 1D.

    """
    if signal.shape != frequencies.shape:
        msg = "y and freq must have the same shape"
        raise ValueError(msg)
    if signal.ndim != 1 or frequencies.ndim != 1:
        msg = "y and freq must be 1-dimensional arrays"
        raise ValueError(msg)
    if signal.size == 0 or frequencies.size == 0:
        return []
    if not thres_abs:
        thres = float(thres * (np.max(signal) - np.min(signal)) + np.min(signal))

    min_dist = int(min_dist)
    dy = np.diff(signal)
    dy = _handle_plateaus(dy)

    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(signal, thres)),
    )[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(signal[peaks])][::-1]
        rem = np.ones(signal.size, dtype=bool)
        rem[peaks] = False
        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False
        peaks = np.arange(signal.size)[~rem]

    return [
        Peak(frequency=float(frequencies[i]), amplitude=float(signal[i]), idx=i)
        for i in peaks
    ]


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
