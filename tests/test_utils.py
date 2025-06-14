import numpy as np

from pyzfn import utils


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
