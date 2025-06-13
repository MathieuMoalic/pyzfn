import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")

from pyzfn import Pyzfn


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
    # Set attributes
    base_sim.attrs["dx"] = 1e-9
    base_sim.attrs["dy"] = 1e-9

    # Add FFT frequency and spectrum data
    freqs = np.linspace(0, 50, 500)
    spec = np.exp(-((freqs - 20) ** 2) / (2 * 2.0**2))  # A Gaussian peak at 20
    spec = np.stack([spec for _ in range(3)], axis=-1)  # shape: (500, 3)
    base_sim.add_ndarray("fft/m/freqs", freqs)
    base_sim.add_ndarray("fft/m/spec", spec)

    # Add modes data and frequency
    modes = np.random.rand(1, 1, 64, 64, 3) + 1j * np.random.rand(1, 1, 64, 64, 3)
    mode_freqs = np.array([20.0])
    base_sim.create_group("modes/m")
    base_sim.add_ndarray("modes/m/arr", modes)
    base_sim.add_ndarray("modes/m/freqs", mode_freqs)

    return base_sim


def test_inner_ispec_runs(sim: Pyzfn) -> None:
    # Should not raise any exceptions and should draw the plot
    sim.ispec(dset_str="m", thres=0.05, fmin=0, fmax=30, log=False)


def test_inner_ispec_log_scale(sim: Pyzfn) -> None:
    # Test with log scale enabled
    sim.ispec(dset_str="m", log=True)


def test_inner_ispec_missing_dataset_raises(sim: Pyzfn) -> None:
    # Remove the expected FFT dataset
    del sim["fft/m/spec"]
    with pytest.raises(KeyError):
        sim.ispec(dset_str="m")


def test_inner_ispec_invalid_attrs(sim: Pyzfn) -> None:
    sim.attrs["dx"] = "not_a_float"
    with pytest.raises(ValueError):
        sim.ispec(dset_str="m")
