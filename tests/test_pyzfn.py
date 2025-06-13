import pytest
import numpy as np
import zarr
from pyzfn import Pyzfn


def test_add_ndarray_valid(base_sim: Pyzfn) -> None:
    data = np.random.rand(4, 4)
    arr = base_sim.add_ndarray("data", data)
    assert isinstance(arr, zarr.Array)
    np.testing.assert_array_equal(arr[:], data)


def test_add_ndarray_overwrite(base_sim: Pyzfn) -> None:
    data1 = np.ones((2, 2))
    data2 = np.zeros((2, 2))
    base_sim.add_ndarray("overwrite_test", data1)
    base_sim.add_ndarray("overwrite_test", data2, overwrite=True)
    np.testing.assert_array_equal(base_sim.get_array("overwrite_test"), data2)


def test_add_ndarray_no_overwrite_raises(base_sim: Pyzfn) -> None:
    base_sim.add_ndarray("foo", np.array([[1]]))
    with pytest.raises(ValueError):
        base_sim.add_ndarray("foo", np.array([[2]]), overwrite=False)
