# flake8: noqa: PLR0913
"""Functions for visualizing and plotting snapshots of Pyzfn datasets."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyzfn import Pyzfn

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .utils import hsl2rgb


def _vector_field_to_rgb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alphas = -np.abs(w) + 1
    hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float64)
    hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2
    hsl[:, :, 1] = np.sqrt(u**2 + v**2 + w**2)
    hsl[:, :, 2] = (w + 1) / 2
    rgb = hsl2rgb(hsl)
    return rgb, alphas


def _create_quiver_grid_and_scale(
    *,
    ax: Axes,
    alphas: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    dy: float,
) -> None:
    stepx = max(int(u.shape[1] / 20), 1)
    stepy = max(int(u.shape[0] / 20), 1)
    scale = 1 / max(stepx, stepy)

    x, y = np.meshgrid(
        np.arange(0, u.shape[1], stepx, dtype=np.float32) * dx * 1e9,
        np.arange(0, u.shape[0], stepy, dtype=np.float32) * dy * 1e9,
    )
    ax.quiver(
        x,
        y,
        u[::stepy, ::stepx],
        v[::stepy, ::stepx],
        alpha=alphas[::stepy, ::stepx],
        angles="xy",
        scale_units="xy",
        scale=scale,
    )


def inner_snapshot(
    self: "Pyzfn",
    dset_str: str = "m",
    *,
    z: int = 0,
    t: int = -1,
    ax: Axes | None = None,
    repeat: int = 1,
) -> Axes:
    """Plot a snapshot of a vector field dataset as an RGB image with quiver overlay.

    Parameters
    ----------
    self : Pyzfn
        The Pyzfn dataset instance.
    dset_str : str, optional
        The dataset string key to visualize (default is "m").
    z : int, optional
        The z-index to plot (default is 0).
    t : int, optional
        The time index to plot (default is -1, last frame).
    ax : Axes or None, optional
        Matplotlib Axes to plot on. If None, a new figure and axes are created.
    repeat : int, optional
        Number of times to tile the image in both dimensions (default is 1).

    Returns
    -------
    Axes
        The matplotlib Axes with the plot.

    Raises
    ------
    TypeError
        If `dx` or `dy` are not floats.

    """
    dx, dy = self.attrs["dx"], self.attrs["dy"]
    if not isinstance(dx, float) or not isinstance(dy, float):
        msg = "dx and dy must be floats"
        raise TypeError(msg)

    m_original = np.array(self.get_array(dset_str)[t, z], dtype=np.float32)

    if ax is None:
        shape_ratio = m_original.shape[1] / m_original.shape[0]
        _, ax = plt.subplots(1, 1, figsize=(3 * shape_ratio, 3), dpi=100)

    m = np.tile(m_original, (repeat, repeat, 1))
    u, v, w = m[:, :, 0], m[:, :, 1], m[:, :, 2]

    rgb, alphas = _vector_field_to_rgb(u, v, w)
    _create_quiver_grid_and_scale(
        ax=ax,
        alphas=alphas,
        u=u,
        v=v,
        dx=dx,
        dy=dy,
    )

    ax.imshow(
        rgb,
        interpolation="None",
        origin="lower",
        aspect="equal",
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        extent=(
            0,
            rgb.shape[1] * dx * 1e9,
            0,
            rgb.shape[0] * dy * 1e9,
        ),
    )
    ax.set(title=self.name, xlabel="x (nm)", ylabel="y (nm)")
    return ax
