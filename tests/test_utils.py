"""Unit tests for pyzfn.utils module.

This module tests utility functions such as color conversions, byte formatting,
and peak finding algorithms provided by pyzfn.utils.
"""

import colorsys
from collections.abc import Sequence

import matplotlib.colors as mcolors
import numpy as np
import pytest
from numpy.typing import NDArray

from pyzfn import utils
from pyzfn.utils import Peak, find_peaks


def _rgba8_to_unit(arr: Sequence[int] | Sequence[float]) -> NDArray[np.float64]:
    """Divide an RGBA tuple in 0-255 range by 256.

    This allows the tuple to be compared directly with the output of ``make_cmap``.

    Returns:
        A NumPy array of floats representing the RGBA values scaled to [0, 1).

    """
    return np.asarray(arr, dtype=float) / 256.0


MIN = (0, 0, 0, 255)
MID = (128, 0, 128, 255)  # purple
MAX = (255, 255, 255, 255)


def test_make_cmap_end_points_and_shape() -> None:
    """Test that make_cmap returns a colormap with correct end points and shape."""
    cmap: mcolors.ListedColormap = utils.make_cmap(MIN, MAX)
    assert isinstance(cmap, mcolors.ListedColormap)
    colours = np.asarray(cmap.colors)
    # basic shape/dtype checks
    assert colours.shape == (256, 4)
    assert colours.dtype == float
    # end points must match min/max (within scaling tolerance)
    np.testing.assert_allclose(colours[0], _rgba8_to_unit(MIN), rtol=0, atol=1e-6)
    np.testing.assert_allclose(colours[-1], _rgba8_to_unit(MAX), rtol=0, atol=1e-6)
    # values should be monotonically increasing in every channel
    tolerance = 1e-9
    assert np.all(np.diff(colours, axis=0) >= -tolerance)


def test_make_cmap_with_midpoint_and_transparency() -> None:
    """Test that make_cmap correctly handles a midpoint and transparency."""
    cmap = utils.make_cmap(MIN, MAX, MID, transparent_zero=True)
    colours = np.asarray(cmap.colors)

    # Index 127 (end of first half) must equal MID in **all four** channels
    np.testing.assert_allclose(colours[127], _rgba8_to_unit(MID), atol=1e-6)

    # Index 128 (start of second half) → same RGB, but alpha forced to zero
    np.testing.assert_allclose(colours[128, :3], _rgba8_to_unit(MID)[:3], atol=1e-6)
    assert colours[128, 3] == pytest.approx(0.0, abs=1e-10)


def test_hsl2rgb_matches_colorsys_reference() -> None:
    """Test that utils.hsl2rgb matches colorsys.hls_to_rgb for random HSL values."""
    rng = np.random.default_rng(0)
    hsl = rng.random((20, 15, 3))
    rgb_ref = np.empty_like(hsl)
    for i, j in np.ndindex(hsl.shape[:2]):
        h, s, l = hsl[i, j]  # noqa: E741
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        rgb_ref[i, j] = (r, g, b)
    rgb_utils = utils.hsl2rgb(hsl)
    np.testing.assert_allclose(rgb_utils, rgb_ref, atol=1e-6)


def test_rgb_hsl_roundtrip_precision() -> None:
    """Test that utils.rgb2hsl and utils.hsl2rgb roundtrip without significant loss."""
    rng = np.random.default_rng(1)
    rgb = rng.random((10, 12, 3))
    hsl = utils.rgb2hsl(rgb)
    rgb_back = utils.hsl2rgb(hsl)
    # Tolerance chosen to be generous but still expose precision regressions
    np.testing.assert_allclose(rgb_back, rgb, atol=5e-3, rtol=0)


def test_format_bytes_basic() -> None:
    """Test basic functionality of format_bytes for small values."""
    # Test small values that should be in bytes
    assert utils.format_bytes(0) == "0 B", "Expected 0 B"
    assert utils.format_bytes(123) == "123 B", "Expected 123 B"
    assert utils.format_bytes(1023) == "1023 B", "Expected 1023 B"


def test_format_bytes_kib() -> None:
    """Test formatting of bytes in KiB, MiB, GiB, and TiB."""
    # Test values that should be in KiB
    assert utils.format_bytes(1024) == "1.00 KiB", "Expected 1.00 KiB"
    assert utils.format_bytes(2048) == "2.00 KiB", "Expected 2.00 KiB"
    assert utils.format_bytes(5000) == "4.88 KiB", "Expected 4.88 KiB"


def test_format_bytes_mib() -> None:
    """Test formatting of bytes in MiB."""
    # Test values that should be in MiB
    assert utils.format_bytes(1048576) == "1.00 MiB", "Expected 1.00 MiB"
    assert utils.format_bytes(2097152) == "2.00 MiB", "Expected 2.00 MiB"
    assert utils.format_bytes(5000000) == "4.77 MiB", "Expected 4.77 MiB"


def test_format_bytes_gib() -> None:
    """Test formatting of bytes in GiB."""
    # Test values that should be in GiB
    assert utils.format_bytes(1073741824) == "1.00 GiB", "Expected 1.00 GiB"
    assert utils.format_bytes(2147483648) == "2.00 GiB", "Expected 2.00 GiB"
    assert utils.format_bytes(5000000000) == "4.66 GiB", "Expected 4.66 GiB"


def test_format_bytes_tib() -> None:
    """Test formatting of bytes in TiB."""
    # Test values that should be in TiB
    assert utils.format_bytes(1099511627776) == "1.00 TiB", "Expected 1.00 TiB"
    assert utils.format_bytes(2199023255552) == "2.00 TiB", "Expected 2.00 TiB"


def test_format_bytes_large_values() -> None:
    """Test formatting of very large byte values in PiB and EiB."""
    # Test very large values (in TiB and above)
    assert utils.format_bytes(1125899906842624) == "1.00 PiB", "Expected 1.00 PiB"
    assert utils.format_bytes(1152921504606846976) == "1.00 EiB", "Expected 1.00 EiB"


def test_single_peak() -> None:
    """Test finding a single peak in a signal."""
    freq = np.linspace(0, 10, 100, dtype=np.float64)
    signal = np.zeros_like(freq)
    peak_idx = 50
    signal[peak_idx] = 1.0  # One sharp peak

    peaks = find_peaks(freq, signal)
    assert len(peaks) == 1
    assert isinstance(peaks[0], Peak)
    assert peaks[0].idx == peak_idx
    assert peaks[0].frequency == freq[peak_idx]
    assert peaks[0].amplitude == signal[peak_idx]


def test_multiple_peaks() -> None:
    """Test finding multiple peaks in a signal."""
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[20] = 0.5
    signal[50] = 1.0
    signal[80] = 0.8

    peaks = find_peaks(freq, signal, thres=0.3, min_dist=10)
    peak_count = 3
    assert len(peaks) == peak_count
    idxs = sorted(p.idx for p in peaks)
    assert idxs == [20, 50, 80]


def test_threshold_relative() -> None:
    """Test finding peaks with a relative threshold."""
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[30] = 0.2
    peak_idx = 70
    signal[peak_idx] = 0.9

    # Use thres=0.5 relative => 0.5*(0.9-0) = 0.45, so only peak[peak_idx] passes
    peaks = find_peaks(freq, signal, thres=0.5)
    assert len(peaks) == 1
    assert peaks[0].idx == peak_idx


def test_threshold_absolute() -> None:
    """Test finding peaks with an absolute threshold."""
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[30] = 0.2
    peak_idx = 70
    signal[peak_idx] = 0.9

    # Use thres_abs=True, thres=0.85 should pass only the second peak
    peaks = find_peaks(freq, signal, thres=0.85, thres_abs=True)
    assert len(peaks) == 1
    assert peaks[0].idx == peak_idx

    # Raise threshold higher: should exclude all
    peaks = find_peaks(freq, signal, thres=1.0, thres_abs=True)
    assert len(peaks) == 0


def test_plateau_peak() -> None:
    """Test finding a peak with a flat top (plateau)."""
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    peak_idx_start = 45
    peak_idx_end = 48
    signal[peak_idx_start:peak_idx_end] = 1.0  # Flat

    peaks = find_peaks(freq, signal)
    assert len(peaks) == 1
    # Any point in plateau is acceptable
    assert peak_idx_start <= peaks[0].idx <= peak_idx_end - 1


def test_min_distance_enforcement() -> None:
    """Test that min_dist correctly enforces minimum distance between peaks."""
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    peak_idx = 30
    signal[peak_idx] = 1.0
    signal[32] = 0.9  # Close neighbor peak

    peaks = find_peaks(freq, signal, min_dist=5)
    assert len(peaks) == 1
    assert peaks[0].idx == peak_idx  # Higher of the two should be kept


def test_empty_input() -> None:
    """Test behavior with empty input arrays."""
    freq = np.array([])
    signal = np.array([])
    peaks = find_peaks(freq, signal)
    assert peaks == []


def test_invalid_shape() -> None:
    """Test that find_peaks raises ValueError for mismatched input shapes."""
    freq = np.linspace(0, 10, 100)
    signal = np.linspace(0, 10, 99)
    with pytest.raises(ValueError, match="same shape"):
        find_peaks(freq, signal)


def test_invalid_ndim() -> None:
    """Test that find_peaks raises ValueError for non-1D inputs."""
    freq = np.ones((100, 2))
    signal = np.ones((100, 2))
    with pytest.raises(ValueError, match="y and freq must be 1-dimensional arrays"):
        find_peaks(freq, signal)
