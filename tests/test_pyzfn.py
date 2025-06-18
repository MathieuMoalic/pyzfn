from numpy.typing import NDArray
import pytest
import numpy as np
import zarr
from pyzfn import Pyzfn
from pathlib import Path


def test_path_not_exist(tmp_path: Path) -> None:
    path = tmp_path / "test_store.zarr"
    if path.exists():
        path.unlink()
    with pytest.raises(FileNotFoundError):
        Pyzfn(path)


def test_path_not_dir(tmp_path: Path) -> None:
    path = tmp_path / "test_store.zarr"
    if path.exists():
        path.unlink()
    path.touch()
    with pytest.raises(NotADirectoryError):
        Pyzfn(path)


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
    result: NDArray[np.float64] = base_sim.g("g_full")
    np.testing.assert_array_equal(result, data)
    assert result.dtype == data.dtype


def test_g_single_slice(base_sim: Pyzfn) -> None:
    data = np.random.rand(10, 10)
    base_sim.add_ndarray("g_slice", data)
    result: NDArray[np.float64] = base_sim.g("g_slice", slices=np.s_[0:5])
    np.testing.assert_array_equal(result, data[slice(0, 5)])
    assert result.shape[0] == 5


def test_g_multi_slice(base_sim: Pyzfn) -> None:
    data = np.arange(100).reshape(10, 10)
    base_sim.add_ndarray("g_multi", data)
    result: NDArray[np.float64] = base_sim.g("g_multi", slices=np.s_[1:5, 2:6])
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
    result: NDArray[np.float64] = base_sim.g("g_sequence", slices=np.s_[[0, 1], 1:4])
    expected = data[[0, 1], 1:4]
    np.testing.assert_array_equal(result, expected)


def test_get_mode_valid(base_sim: Pyzfn) -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    modes = np.random.rand(3, 4, 4) + 1j * np.random.rand(3, 4, 4)
    modes = modes.astype(np.complex64)

    base_sim.add_ndarray("modes/test/freqs", freqs)
    base_sim.add_ndarray("modes/test/arr", modes)

    f = 2.1  # Closest is index 1 (2.0)
    result = base_sim.get_mode("test", f)

    np.testing.assert_array_equal(result, modes[1])
    assert result.dtype == np.complex64


def test_get_mode_missing_freqs(base_sim: Pyzfn) -> None:
    with pytest.raises(KeyError, match="modes/test/freqs"):
        base_sim.get_mode("test", 1.0)


def test_get_mode_missing_arr(base_sim: Pyzfn) -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    base_sim.add_ndarray("modes/test/freqs", freqs)

    with pytest.raises(KeyError, match="modes/test/arr"):
        base_sim.get_mode("test", 2.0)


def test_get_mode_closest_freq_selection(base_sim: Pyzfn) -> None:
    freqs = np.array([1.0, 2.0, 3.0])
    modes = np.array(
        [np.full((2, 2), 10 + 1j), np.full((2, 2), 20 + 2j), np.full((2, 2), 30 + 3j)],
        dtype=np.complex64,
    )

    base_sim.add_ndarray("modes/sample/freqs", freqs)
    base_sim.add_ndarray("modes/sample/arr", modes)

    result = base_sim.get_mode("sample", 2.6)  # Closest is 3.0
    expected = modes[2]

    np.testing.assert_array_equal(result, expected)


def test_get_mode_returns_copy(base_sim: Pyzfn) -> None:
    freqs = np.array([5.0])
    mode = np.array([[1 + 1j]], dtype=np.complex64)
    base_sim.add_ndarray("modes/check/freqs", freqs)
    base_sim.add_ndarray("modes/check/arr", mode[np.newaxis, ...])

    result = base_sim.get_mode("check", 5.0)
    result[0, 0] = 0 + 0j

    # Check original remains unchanged
    original = base_sim.get_array("modes/check/arr")[0]
    assert not np.allclose(original, result)


def test_repr(base_sim: Pyzfn) -> None:
    assert "Pyzfn" in repr(base_sim)


def test_str(base_sim: Pyzfn) -> None:
    assert "Pyzfn" in str(base_sim)


def test_p_prints_tree(base_sim: Pyzfn, capsys: pytest.CaptureFixture[str]) -> None:
    """
    .p writes a Rich tree to stdout and returns None.
    We simply capture the console and make sure the group name shows up.
    """
    base_sim.add_ndarray("modes/check/freqs", np.array([5.0]))
    _ = base_sim.p  # prints to console
    captured = capsys.readouterr()
    # Rich adds ANSI codes; look for the plain group name.
    assert base_sim.name in captured.out
    # .p is a property that should not accidentally mutate state
    assert "Tree(" not in repr(base_sim)  # sanity: nothing strange leaked


def test_rm_removes_dataset(base_sim: Pyzfn) -> None:
    data = np.arange(4)
    base_sim.add_ndarray("to_delete", data)
    # After removal the key must disappear
    base_sim.rm("to_delete")
    with pytest.raises(KeyError):
        base_sim.get_array("to_delete")


def test_rm_missing_key_raises(base_sim: Pyzfn) -> None:
    with pytest.raises(KeyError):
        base_sim.rm("i_do_not_exist")


def test_get_array_rejects_group(base_sim: Pyzfn) -> None:
    base_sim.create_group("grp")
    with pytest.raises(ValueError, match="must be a zarr array"):
        base_sim.get_array("grp")
    # keep fixture clean – remove subgroup again
    del base_sim["grp"]
