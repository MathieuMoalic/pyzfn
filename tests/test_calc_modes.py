"""Tests for the calc_modes functionality in the pyzfn package."""

import numpy as np
import pytest
from numpy import s_ as s

from pyzfn import Pyzfn

SHAPE = (64, 2, 1, 4, 3)  # Example shape for the dataset


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
    """Fixture that initializes a Pyzfn simulation.

    Returns:
        Pyzfn: The initialized simulation object with random data and attributes.

    """
    rng = np.random.default_rng(42)
    data = rng.standard_normal(SHAPE, dtype=np.float32)
    dset = base_sim.add_ndarray("m", data=data)
    dset.attrs["t"] = list(np.arange(0, SHAPE[0], dtype=float))
    base_sim.attrs.update({"dx": 1.0, "dy": 1.0, "dz": 1.0})
    return base_sim


def test_calc_modes_basic(sim: Pyzfn) -> None:
    """Test basic functionality of calc_modes and output array shapes/dtypes."""
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
    """Test that windowing and no windowing produce different results in calc_modes."""
    sim.calc_modes("m", dset_out_str="win", slices=s[1:], window=True)
    sim.calc_modes("m", dset_out_str="nowin", slices=s[1:], window=False)
    spec_win = sim.get_array("fft/win/spec")[:]
    spec_nowin = sim.get_array("fft/nowin/spec")[:]
    assert not np.allclose(spec_win, spec_nowin), "Windowing has no effect"


def test_calc_modes_missing_time_attr(sim: Pyzfn) -> None:
    """Test that calc_modes raises ValueError when the time attribute is missing."""
    del sim["m"].attrs["t"]
    with pytest.raises(ValueError, match="lacks required time attribute"):
        sim.calc_modes("m", dset_out_str="no_t")


def test_calc_modes_wrong_rank(sim: Pyzfn) -> None:
    """Test that calc_modes raises ValueError."""
    bad = sim.add_ndarray("bad", data=np.ones((10, 10), dtype=np.float32))
    bad.attrs["t"] = list(range(10))
    with pytest.raises(ValueError, match="Expected a 5-D array"):
        sim.calc_modes("bad")


def test_calc_modes_partial_slice(sim: Pyzfn) -> None:
    """Test calc_modes with a partial slice of the input array."""
    sim.calc_modes("m", dset_out_str="slice", slices=s[10:20, 0:1])
    arr = sim.get_array("modes/slice/arr")
    assert arr.shape == (
        6,
        1,
        SHAPE[2],
        SHAPE[3],
        SHAPE[4],
    )  # rfft of 10 frames -> 6 frequency bins


MIN_TIME_FRAMES = 2


def test_calc_modes_min_time_frames(sim: Pyzfn) -> None:
    """Test calc_modes with the minimum number of time frames (2)."""
    sim.calc_modes("m", dset_out_str="short", slices=s[0:MIN_TIME_FRAMES])
    arr = sim.get_array("modes/short/arr")
    assert arr.shape[0] == MIN_TIME_FRAMES  # rfft(2) -> 2 frequencies


def test_calc_modes_time_attr_mismatch(sim: Pyzfn) -> None:
    """Test that calc_modes raises ValueError when with wrong length."""
    sim["m"].attrs["t"] = list(range(10))  # mismatched time length
    with pytest.raises(ValueError, match="does not match time dimension"):
        sim.calc_modes("m")


def test_calc_modes_dtype_conversion(sim: Pyzfn) -> None:
    """Test that calc_modes preserves or converts dtype to np.complex64 as expected."""
    sim.get_array("m")[:] = np.array(sim.get_array("m")[:], dtype=np.complex64)
    sim.calc_modes("m", dset_out_str="dtype_test")
    assert sim.get_array("modes/dtype_test/arr").dtype == np.complex64


def test_calc_modes_windowing_energy_change(sim: Pyzfn) -> None:
    """Test that windowing changes the energy spectrum in calc_modes."""
    sim.calc_modes("m", dset_out_str="win_energy", window=True)
    sim.calc_modes("m", dset_out_str="no_win_energy", window=False)

    spec_win = sim.get_array("fft/win_energy/sum")[:]
    spec_nowin = sim.get_array("fft/no_win_energy/sum")[:]
    assert not np.allclose(spec_win, spec_nowin)
