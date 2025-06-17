import numpy as np
import pytest
from pyzfn import utils
from pyzfn.utils import find_peaks, Peak


def test_make_cmap() -> None:
    min_color = (255, 33, 53, 255)
    max_color = (139, 233, 253, 255)
    mid_color = (13, 33, 253, 255)
    utils.make_cmap(min_color, max_color, mid_color)
    utils.make_cmap(min_color, max_color, mid_color, transparent_zero=True)
    utils.make_cmap(min_color, max_color, transparent_zero=True)


def test_hsl2rgb() -> None:
    hsl = np.ones((40, 50, 3), dtype=np.float64)
    utils.hsl2rgb(hsl)


def test_format_bytes_basic() -> None:
    # Test small values that should be in bytes
    assert utils.format_bytes(0) == "0 B", "Expected 0 B"
    assert utils.format_bytes(123) == "123 B", "Expected 123 B"
    assert utils.format_bytes(1023) == "1023 B", "Expected 1023 B"


def test_format_bytes_kib() -> None:
    # Test values that should be in KiB
    assert utils.format_bytes(1024) == "1.00 KiB", "Expected 1.00 KiB"
    assert utils.format_bytes(2048) == "2.00 KiB", "Expected 2.00 KiB"
    assert utils.format_bytes(5000) == "4.88 KiB", "Expected 4.88 KiB"


def test_format_bytes_mib() -> None:
    # Test values that should be in MiB
    assert utils.format_bytes(1048576) == "1.00 MiB", "Expected 1.00 MiB"
    assert utils.format_bytes(2097152) == "2.00 MiB", "Expected 2.00 MiB"
    assert utils.format_bytes(5000000) == "4.77 MiB", "Expected 4.77 MiB"


def test_format_bytes_gib() -> None:
    # Test values that should be in GiB
    assert utils.format_bytes(1073741824) == "1.00 GiB", "Expected 1.00 GiB"
    assert utils.format_bytes(2147483648) == "2.00 GiB", "Expected 2.00 GiB"
    assert utils.format_bytes(5000000000) == "4.66 GiB", "Expected 4.66 GiB"


def test_format_bytes_tib() -> None:
    # Test values that should be in TiB
    assert utils.format_bytes(1099511627776) == "1.00 TiB", "Expected 1.00 TiB"
    assert utils.format_bytes(2199023255552) == "2.00 TiB", "Expected 2.00 TiB"


def test_format_bytes_large_values() -> None:
    # Test very large values (in TiB and above)
    assert utils.format_bytes(1125899906842624) == "1.00 PiB", "Expected 1.00 PiB"
    assert utils.format_bytes(1152921504606846976) == "1.00 EiB", "Expected 1.00 EiB"


def test_single_peak():
    freq = np.linspace(0, 10, 100, dtype=np.float64)
    signal = np.zeros_like(freq)
    signal[50] = 1.0  # One sharp peak

    peaks = find_peaks(freq, signal)
    assert len(peaks) == 1
    assert isinstance(peaks[0], Peak)
    assert peaks[0].idx == 50
    assert peaks[0].frequency == freq[50]
    assert peaks[0].amplitude == signal[50]


def test_multiple_peaks():
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[20] = 0.5
    signal[50] = 1.0
    signal[80] = 0.8

    peaks = find_peaks(freq, signal, thres=0.3, min_dist=10)
    assert len(peaks) == 3
    idxs = sorted(p.idx for p in peaks)
    assert idxs == [20, 50, 80]


def test_threshold_relative():
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[30] = 0.2
    signal[70] = 0.9

    # Use thres=0.5 relative => 0.5*(0.9-0) = 0.45, so only peak[70] passes
    peaks = find_peaks(freq, signal, thres=0.5)
    assert len(peaks) == 1
    assert peaks[0].idx == 70


def test_threshold_absolute():
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[30] = 0.2
    signal[70] = 0.9

    # Use thres_abs=True, thres=0.85 should pass only the second peak
    peaks = find_peaks(freq, signal, thres=0.85, thres_abs=True)
    assert len(peaks) == 1
    assert peaks[0].idx == 70

    # Raise threshold higher: should exclude all
    peaks = find_peaks(freq, signal, thres=1.0, thres_abs=True)
    assert len(peaks) == 0


def test_plateau_peak():
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[45:48] = 1.0  # Flat-topped peak

    peaks = find_peaks(freq, signal)
    assert len(peaks) == 1
    assert 45 <= peaks[0].idx <= 47  # Any point in plateau is acceptable


def test_min_distance_enforcement():
    freq = np.linspace(0, 10, 100)
    signal = np.zeros_like(freq)
    signal[30] = 1.0
    signal[32] = 0.9  # Close neighbor peak

    peaks = find_peaks(freq, signal, min_dist=5)
    assert len(peaks) == 1
    assert peaks[0].idx == 30  # Higher of the two should be kept


def test_empty_input():
    freq = np.array([])
    signal = np.array([])
    peaks = find_peaks(freq, signal)
    assert peaks == []


def test_invalid_shape():
    freq = np.linspace(0, 10, 100)
    signal = np.linspace(0, 10, 99)
    with pytest.raises(ValueError):
        find_peaks(freq, signal)


def test_invalid_ndim():
    freq = np.ones((100, 2))
    signal = np.ones((100, 2))
    with pytest.raises(ValueError):
        find_peaks(freq, signal)
