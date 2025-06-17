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


def test_g_full_slice(base_sim: Pyzfn) -> None:
    data = np.random.rand(3, 4, 5)
    base_sim.add_ndarray("g_full", data)
    result = base_sim.g("g_full")
    np.testing.assert_array_equal(result, data)
    assert result.dtype == data.dtype


def test_g_single_slice(base_sim: Pyzfn) -> None:
    data = np.random.rand(10, 10)
    base_sim.add_ndarray("g_slice", data)
    result = base_sim.g("g_slice", slices=np.s_[0:5])
    np.testing.assert_array_equal(result, data[slice(0, 5)])
    assert result.shape[0] == 5


def test_g_multi_slice(base_sim: Pyzfn) -> None:
    data = np.arange(100).reshape(10, 10)
    base_sim.add_ndarray("g_multi", data)
    result = base_sim.g("g_multi", slices=np.s_[1:5, 2:6])
    np.testing.assert_array_equal(result, data[1:5, 2:6])


def test_g_dataset_not_found_raises(base_sim: Pyzfn) -> None:
    with pytest.raises(KeyError):
        base_sim.g("nonexistent_dataset")


def test_g_too_many_slices(base_sim: Pyzfn) -> None:
    data = np.random.rand(3, 3)
    base_sim.add_ndarray("g_shape_check", data)
    with pytest.raises(ValueError, match="Too many slices"):
        # np.s_ generates a slice tuple: equivalent to (slice(None),)*5
        base_sim.g("g_shape_check", slices=np.s_[:, :, :, :, :])


def test_g_sequency_indexing(base_sim: Pyzfn) -> None:
    data = np.random.rand(5, 5)
    base_sim.add_ndarray("g_sequence", data)
    result = base_sim.g("g_sequence", slices=np.s_[[0, 1], 1:4])
    expected = data[[0, 1], 1:4]
    np.testing.assert_array_equal(result, expected)
