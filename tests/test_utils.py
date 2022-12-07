from pyzfn import utils, Pyzfn
import numpy as np
import pytest
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
