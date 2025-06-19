"""Tests for the snapshot functionality of the Pyzfn module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from pyzfn import Pyzfn


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
    """Fixture that creates a dummy Pyzfn simulation.

    Returns:
        Pyzfn: The simulation object with a dummy magnetisation dataset and attributes.

    """
    # Create a dummy magnetisation dataset 'm'
    # Shape: (T, Z, Y, X, 3) — consistent with access pattern (t, z, y, x, comp)
    shape = (1, 1, 4, 4, 3)
    data = np.zeros(shape, dtype=np.float32)
    data[0, 0, :, :, 0] = 1.0  # u component
    data[0, 0, :, :, 1] = 0.0  # v component
    data[0, 0, :, :, 2] = 0.0  # w component
    base_sim.attrs["dx"] = 1e-9  # Set dx attribute
    base_sim.attrs["dy"] = 1e-9  # Set dy attribute

    base_sim.add_ndarray("m", data)

    return base_sim


def test_snapshot_returns_axes(sim: Pyzfn) -> None:
    """Test that snapshot returns an Axes object."""
    ax = sim.snapshot()
    assert isinstance(ax, Axes)


def test_snapshot_with_zero_frame(sim: Pyzfn) -> None:
    """Test that snapshot works with a zero frame."""
    ax = sim.snapshot()
    assert isinstance(ax, Axes)


def test_snapshot_with_repeat(sim: Pyzfn) -> None:
    """Test that snapshot works with a repeat parameter."""
    ax = sim.snapshot(repeat=2)
    assert isinstance(ax, Axes)


def test_snapshot_with_custom_ax(sim: Pyzfn) -> None:
    """Test that snapshot can use a custom Axes object."""
    _, ax_in = plt.subplots()
    ax_out = sim.snapshot(ax=ax_in)
    assert ax_out is ax_in


def test_snapshot_dx_error(sim: Pyzfn) -> None:
    """Test that snapshot raises TypeError if dx is not a float."""
    sim.attrs["dx"] = "not_a_float"
    with pytest.raises(TypeError, match="dx and dy must be floats"):
        sim.snapshot()
