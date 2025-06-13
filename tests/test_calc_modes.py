import numpy as np
from numpy import s_ as s
import pytest
from pyzfn import Pyzfn

SHAPE = (64, 2, 1, 4, 3)  # Example shape for the dataset


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
    rng = np.random.default_rng(42)
    data = rng.standard_normal(SHAPE, dtype=np.float32)
    dset = base_sim.add_ndarray("m", data=data)
    dset.attrs["t"] = list(np.arange(0, SHAPE[0], dtype=float))
    base_sim.attrs.update({"dx": 1.0, "dy": 1.0, "dz": 1.0})
    return base_sim


def test_calc_modes_basic(sim: Pyzfn) -> None:
    sim.calc_modes("m")
    assert sim.get_array("fft/m/freqs").shape == (SHAPE[0] // 2 + 1,)
    assert sim.get_array("fft/m/freqs").dtype == np.float64

    assert sim.get_array("fft/m/spec").shape == (SHAPE[0] // 2 + 1, 3)
    assert sim.get_array("fft/m/spec").dtype == np.float32

    assert sim.get_array("fft/m/sum").shape == (SHAPE[0] // 2 + 1, 3)
    assert sim.get_array("fft/m/sum").dtype == np.float32

    assert sim.get_array("modes/m/freqs").shape == (SHAPE[0] // 2 + 1,)
    assert sim.get_array("modes/m/freqs").dtype == np.float64

    assert sim.get_array("modes/m/arr").shape == (
        SHAPE[0] // 2 + 1,
        SHAPE[1],
        SHAPE[2],
        SHAPE[3],
        SHAPE[4],
    )
    assert sim.get_array("modes/m/arr").dtype == np.complex64


def test_calc_modes_nowindow_differs(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="win", slices=s[1:], window=True)
    sim.calc_modes("m", dset_out_str="nowin", slices=s[1:], window=False)
    spec_win = sim.get_array("fft/win/spec")[:]
    spec_nowin = sim.get_array("fft/nowin/spec")[:]
    assert not np.allclose(spec_win, spec_nowin), "Windowing has no effect"


def test_calc_modes_overwrite_guard(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="safe")
    with pytest.raises(FileExistsError):
        sim.calc_modes("m", dset_out_str="safe", overwrite=False)


def test_calc_modes_missing_time_attr(sim: Pyzfn) -> None:
    del sim["m"].attrs["t"]
    with pytest.raises(ValueError):
        sim.calc_modes("m", dset_out_str="no_t")


def test_calc_modes_wrong_rank(sim: Pyzfn) -> None:
    bad = sim.add_ndarray("bad", data=np.ones((10, 10), dtype=np.float32))
    bad.attrs["t"] = list(range(10))
    with pytest.raises(ValueError):
        sim.calc_modes("bad")


def test_calc_modes_partial_slice(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="slice", slices=s[10:20, 0:1])
    arr = sim.get_array("modes/slice/arr")
    assert arr.shape == (
        6,
        1,
        SHAPE[2],
        SHAPE[3],
        SHAPE[4],
    )  # rfft of 10 frames -> 6 frequency bins


def test_calc_modes_min_time_frames(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="short", slices=s[0:2])
    arr = sim.get_array("modes/short/arr")
    assert arr.shape[0] == 2  # rfft(2) -> 2 frequencies


def test_calc_modes_time_attr_mismatch(sim: Pyzfn) -> None:
    sim["m"].attrs["t"] = list(range(10))  # mismatched time length
    with pytest.raises(ValueError, match="does not match time dimension"):
        sim.calc_modes("m")


def test_calc_modes_dtype_conversion(sim: Pyzfn) -> None:
    sim.get_array("m")[:] = np.array(sim.get_array("m")[:], dtype=np.complex64)
    sim.calc_modes("m", dset_out_str="dtype_test")
    assert sim.get_array("modes/dtype_test/arr").dtype == np.complex64


def test_calc_modes_windowing_energy_change(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="win_energy", window=True)
    sim.calc_modes("m", dset_out_str="no_win_energy", window=False)

    spec_win = sim.get_array("fft/win_energy/sum")[:]
    spec_nowin = sim.get_array("fft/no_win_energy/sum")[:]
    assert not np.allclose(spec_win, spec_nowin)


def test_calc_modes_explicit_overwrite(sim: Pyzfn) -> None:
    sim.calc_modes("m", dset_out_str="overwrite_test")
    sim.calc_modes(
        "m", dset_out_str="overwrite_test", overwrite=True
    )  # should not raise


def test_calc_modes_shapes(sim: Pyzfn) -> None:
    sim.calc_modes("m")
    sim.calc_modes(
        "m", dset_out_str="overwrite_test", overwrite=True
    )  # should not raise
