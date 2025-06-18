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
    transparent_zero: bool = False,
) -> mcolors.ListedColormap:
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
    rgb = np.clip(rgb, 0, 1)
    return rgb


def rgb2hsl(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
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
    frequency: float
    amplitude: float
    idx: int


Numeric = np.integer | np.floating


def find_peaks(
    frequencies: NDArray[Numeric],
    signal: NDArray[Numeric],
    thres: float = 0.3,
    min_dist: int = 1,
    thres_abs: bool = False,
) -> list[Peak]:
    if signal.shape != frequencies.shape:
        raise ValueError("y and freq must have the same shape")
    if signal.ndim != 1 or frequencies.ndim != 1:
        raise ValueError("y and freq must be 1-dimensional arrays")
    if signal.size == 0 or frequencies.size == 0:
        return []
    if not thres_abs:
        thres = float(thres * (np.max(signal) - np.min(signal)) + np.min(signal))

    min_dist = int(min_dist)
    dy = np.diff(signal)

    zeros = np.where(dy == 0)[0]

    if len(zeros):
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

    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(signal, thres))
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


def format_bytes(byte_size: int) -> str:
    if byte_size < 1024:
        return f"{byte_size} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    size = float(byte_size)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"
