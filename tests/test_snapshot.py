import numpy as np
import pytest
from matplotlib.axes import Axes
from pyzfn import Pyzfn


@pytest.fixture
def sim(base_sim: Pyzfn) -> Pyzfn:
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
    ax = sim.snapshot()
    assert isinstance(ax, Axes)


def test_snapshot_with_zero_frame(sim: Pyzfn) -> None:
    ax = sim.snapshot()
    assert isinstance(ax, Axes)


def test_snapshot_with_repeat(sim: Pyzfn) -> None:
    ax = sim.snapshot(repeat=2)
    assert isinstance(ax, Axes)


def test_snapshot_with_custom_ax(sim: Pyzfn) -> None:
    import matplotlib.pyplot as plt

    _, ax_in = plt.subplots()
    ax_out = sim.snapshot(ax=ax_in)
    assert ax_out is ax_in


def test_snapshot_dx_error(sim: Pyzfn) -> None:
    sim.attrs["dx"] = "not_a_float"
    with pytest.raises(ValueError, match="dx and dy must be floats"):
        sim.snapshot()
