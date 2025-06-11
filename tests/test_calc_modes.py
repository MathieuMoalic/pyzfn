import numpy as np
import pytest
import zarr
from pyzfn import Pyzfn


@pytest.fixture
def sim():
    sim = Pyzfn(zarr.MemoryStore())
    rng = np.random.default_rng(42)
    data = rng.standard_normal((64, 2, 18, 12, 3), dtype=np.float32)
    dset = sim.z.create_dataset("m", data=data)
    dset.attrs["t"] = list(np.arange(0, 64, dtype=np.float32))
    sim.z.attrs.update({"dx": 1.0, "dy": 1.0, "dz": 1.0})
    yield sim
    # shutil.rmtree(path)


def test_calc_modes_basic(sim):
    sim.calc_modes("m")
    assert sim.z["fft/m/freqs"].shape[0] == 33
    assert sim.z["modes/m/arr"].dtype == np.complex64
    assert sim.z["modes/m/arr"].shape == (33, 2, 18, 12, 3)


def test_calc_modes_nowindow_differs(sim):
    sim.calc_modes("m", dset_out_str="win", ymax=4, window=True)
    sim.calc_modes("m", dset_out_str="nowin", ymax=4, window=False)
    spec_win = np.asarray(sim.z["fft/win/spec"])
    spec_nowin = np.asarray(sim.z["fft/nowin/spec"])
    assert not np.allclose(spec_win, spec_nowin), "Windowing has no effect"


def test_calc_modes_overwrite_guard(sim):
    sim.calc_modes("m", dset_out_str="safe")
    with pytest.raises(FileExistsError):
        sim.calc_modes("m", dset_out_str="safe", overwrite=False)


def test_calc_modes_zero_length_and_bounds(sim):
    with pytest.raises(IndexError):
        sim.calc_modes("m", zmax=999)
    with pytest.raises(ValueError):
        sim.calc_modes("m", tmin=10, tmax=10)


def test_calc_modes_missing_time_attr(sim):
    del sim.z["m"].attrs["t"]
    with pytest.raises(ValueError):
        sim.calc_modes("m", dset_out_str="no_t")


def test_calc_modes_wrong_rank(sim):
    bad = sim.z.create_dataset("bad", data=np.ones((10, 10)))
    bad.attrs["t"] = list(range(10))
    with pytest.raises(ValueError):
        sim.calc_modes("bad")


def test_calc_modes_partial_slice(sim):
    sim.calc_modes("m", dset_out_str="slice", tmin=10, tmax=20, zmin=0, zmax=1)
    arr = sim.z["modes/slice/arr"]
    assert arr.shape == (6, 1, 18, 12, 3)  # rfft of 10 frames -> 6 frequency bins


def test_calc_modes_min_time_frames(sim):
    sim.calc_modes("m", dset_out_str="short", tmin=0, tmax=2)
    arr = sim.z["modes/short/arr"]
    assert arr.shape[0] == 2  # rfft(2) -> 2 frequencies


def test_calc_modes_skip_memory_check(sim):
    sim.calc_modes("m", dset_out_str="memskip", skip_memory_check=True)
    assert "memskip" in sim.z["modes"]


def test_calc_modes_time_attr_mismatch(sim):
    sim.z["m"].attrs["t"] = list(range(10))  # mismatched time length
    with pytest.raises(ValueError, match="does not match time dimension"):
        sim.calc_modes("m")


def test_calc_modes_dtype_conversion(sim):
    sim.z["m"][:] = sim.z["m"][:].astype(np.float64)
    sim.calc_modes("m", dset_out_str="dtype_test")
    assert sim.z["modes/dtype_test/arr"].dtype == np.complex64


def test_calc_modes_windowing_energy_change(sim):
    sim.calc_modes("m", dset_out_str="win_energy", window=True)
    sim.calc_modes("m", dset_out_str="no_win_energy", window=False)

    spec_win = np.asarray(sim.z["fft/win_energy/sum"])
    spec_nowin = np.asarray(sim.z["fft/no_win_energy/sum"])
    assert not np.allclose(spec_win, spec_nowin)


def test_calc_modes_explicit_overwrite(sim):
    sim.calc_modes("m", dset_out_str="overwrite_test")
    sim.calc_modes(
        "m", dset_out_str="overwrite_test", overwrite=True
    )  # should not raise
