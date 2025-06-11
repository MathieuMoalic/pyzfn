import pytest
import zarr.storage
from pyzfn import Pyzfn
import numpy as np

import shutil


@pytest.fixture
def sim():
    path = "/tmp/test.zarr"
    sim = Pyzfn(zarr.storage.LocalStore(path))
    yield sim
    shutil.rmtree(path, ignore_errors=True)


def test_basic(sim):
    assert sim.path == "file:///tmp/test.zarr"
    assert sim.clean_path == "/tmp/test.zarr"
    assert sim.name == "test"


def test_repr_and_str(sim):
    assert repr(sim) == "Pyzfn('test')"
    assert str(sim) == "Pyzfn('test')"


def test___name__(sim):
    assert sim.__name__() == "test"


def test_path_property(sim):
    assert sim.path.startswith("file:///tmp/")
    assert sim.path.endswith("test.zarr")


def test_name_property(sim):
    assert sim.name == "test"


def test_p_property(sim):
    sim.add_ndarray("arr", data=np.zeros((3, 3)))
    expected = "test /\n└── arr (3, 3) float64\n"
    assert sim.p == expected


def test_add_ndarray_valid(sim):
    data = np.random.rand(4, 4)
    arr = sim.add_ndarray("data", data)
    assert isinstance(arr, zarr.Array)
    np.testing.assert_array_equal(arr[:], data)


def test_add_ndarray_invalid_type(sim):
    with pytest.raises(TypeError):
        sim.add_ndarray("bad", data=[[1, 2], [3, 4]])


def test_add_ndarray_overwrite(sim):
    data1 = np.ones((2, 2))
    data2 = np.zeros((2, 2))
    sim.add_ndarray("overwrite_test", data1)
    sim.add_ndarray("overwrite_test", data2, overwrite=True)
    np.testing.assert_array_equal(sim["overwrite_test"][:], data2)


def test_add_ndarray_no_overwrite_raises(sim):
    sim.add_ndarray("foo", np.array([[1]]))
    with pytest.raises(ValueError):
        sim.add_ndarray("foo", np.array([[2]]), overwrite=False)
