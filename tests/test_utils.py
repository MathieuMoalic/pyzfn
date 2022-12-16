from pyzfn import utils
import numpy as np

import os


def test_wisdom_name_from_array() -> None:
    arr = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    assert utils.wisdom_name_from_array(arr) == os.path.expanduser(
        "~/.cache/fftw/1_1_1_1_1_float32"
    )


def test_save_wisdom() -> None:
    arr = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    utils.save_wisdom(arr)
    assert os.path.exists(os.path.expanduser("~/.cache/fftw/1_1_1_1_1_float32"))


def test_load_wisdom() -> None:
    arr = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    assert utils.load_wisdom(arr)
    arr = np.ones((1, 1, 1, 1, 2), dtype=np.float32)
    assert utils.load_wisdom(arr) is False


def test_make_cmap() -> None:
    min_color = (255, 33, 53, 255)
    max_color = (139, 233, 253, 255)
    mid_color = (13, 33, 253, 255)
    utils.make_cmap(min_color, max_color, mid_color)
    utils.make_cmap(min_color, max_color, mid_color, transparent_zero=True)
    utils.make_cmap(min_color, max_color, transparent_zero=True)


def test_save_ovf() -> None:
    arr = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    utils.save_ovf("test.ovf", arr)
    os.remove("test.ovf")


def test_load_ovf() -> None:
    arr = np.ones((1, 1, 1, 1, 1), dtype=np.float32)
    utils.save_ovf("test.ovf", arr)
    arr2 = utils.load_ovf("test.ovf")
    os.remove("test.ovf")
    assert arr == arr2


def test_ovf_parms() -> None:
    arr = np.ones((1, 2, 3, 4, 1), dtype=np.float32)
    utils.save_ovf("test.ovf", arr)
    parms = utils.get_ovf_parms("test.ovf")
    os.remove("test.ovf")
    assert parms["comp"] == 1
    assert parms["Nx"] == 3
    assert parms["Ny"] == 2
    assert parms["Nz"] == 1
    assert parms["dx"] == 1e-9
    assert parms["dy"] == 1e-9
    assert parms["dz"] == 1e-9


def test_get_slices() -> None:
    shape = (100, 1, 40, 40, 3)
    chunks = (1, 1, 20, 10, 3)
    slices = (
        slice(None),
        slice(None),
        slice(None),
        slice(None),
        slice(None),
    )
    utils.get_slices(shape, chunks, slices)


def test_load_mpl_style() -> None:
    # with pytest.raises(Exception):
    utils.load_mpl_style()


def test_save_current_mplstyle() -> None:
    utils.save_current_mplstyle("style")
    os.remove("style")


def test_hsl2rgb() -> None:
    hsl = np.ones((40, 50, 3), dtype=np.float32)
    utils.hsl2rgb(hsl)
