"""Tests for the ispec functionality in the Pyzfn package."""

from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

mpl.use("Agg")

from pyzfn import Pyzfn


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
    """Prepare a Pyzfn simulation object with FFT and mode data for testing.

    Parameters
    ----------
    base_sim : Pyzfn
        The base Pyzfn simulation object to modify.

    Returns
    -------
    Pyzfn
        The modified Pyzfn simulation object with test data.

    """
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
    rng = np.random.default_rng()
    modes = rng.random((1, 1, 64, 64, 3)) + 1j * rng.random((1, 1, 64, 64, 3))
    mode_freqs = np.array([20.0])
    base_sim.create_group("modes/m")
    base_sim.add_ndarray("modes/m/arr", modes)
    base_sim.add_ndarray("modes/m/freqs", mode_freqs)

    return base_sim


def test_inner_ispec_runs(sim: Pyzfn) -> None:
    """Test that ispec runs without raising exceptions and draws the plot."""
    # Should not raise any exceptions and should draw the plot
    sim.ispec(dset_str="m", threshold=0.05, fmin=0, fmax=30, log=False)


def test_inner_ispec_log_scale(sim: Pyzfn) -> None:
    """Test that ispec runs with log scale enabled without raising exceptions."""
    # Test with log scale enabled
    sim.ispec(dset_str="m", log=True)


def test_inner_ispec_missing_dataset_raises(sim: Pyzfn) -> None:
    """Test that ispec raises a KeyError when the expected FFT dataset is missing."""
    # Remove the expected FFT dataset
    del sim["fft/m/spec"]
    with pytest.raises(KeyError):
        sim.ispec(dset_str="m")


def test_inner_ispec_invalid_attrs(sim: Pyzfn) -> None:
    """Test that ispec raises a ValueError when dx attribute is not a float."""
    sim.attrs["dx"] = "not_a_float"
    with pytest.raises(TypeError, match="dx and dy must be floats"):
        sim.ispec(dset_str="m")


def test_inner_ispec_onclick_behavior(sim: Pyzfn) -> None:
    """Test that the onclick handler for ispec calls get_mode."""
    onclick, fig, ax_spec, _ = sim.ispec()
    event = MouseEvent(name="button_press_event", canvas=fig.canvas, x=0, y=0)
    event.inaxes = ax_spec
    event.xdata = 20.0
    event.button = MouseButton.LEFT

    with patch.object(sim, "get_mode") as mock_get_mode:
        mock_get_mode.return_value = [np.ones((10, 10, 3), dtype=complex)]
        onclick(event)
        mock_get_mode.assert_called_once()


def test_inner_ispec_onclick_right_behavior(sim: Pyzfn) -> None:
    """Test that the onclick handler for ispec calls get_mode on right mouse button."""
    onclick, fig, ax_spec, _ = sim.ispec()
    event = MouseEvent(name="button_press_event", canvas=fig.canvas, x=0, y=0)
    event.inaxes = ax_spec
    event.xdata = 20.0
    event.button = MouseButton.RIGHT

    with patch.object(sim, "get_mode") as mock_get_mode:
        mock_get_mode.return_value = [np.ones((10, 10, 3), dtype=complex)]
        onclick(event)
        mock_get_mode.assert_called_once()


def test_inner_ispec_onclick_none_behavior(sim: Pyzfn) -> None:
    """Test that the onclick handler does not call get_mode when xdata is None."""
    onclick, fig, ax_spec, _ = sim.ispec()
    event = MouseEvent(name="button_press_event", canvas=fig.canvas, x=0, y=0)
    event.inaxes = ax_spec
    event.xdata = None
    event.button = MouseButton.RIGHT

    with patch.object(sim, "get_mode") as mock_get_mode:
        mock_get_mode.return_value = [np.ones((10, 10, 3), dtype=complex)]
        onclick(event)
        mock_get_mode.assert_not_called()
