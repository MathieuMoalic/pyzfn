"""Tests for the pyzfn.ovf module."""

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from pyzfn import ovf


def make_array(
    nz: int,
    ny: int,
    nx: int,
    ncomp: int,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Deterministic pseudo-random array in the exact dtype used by save_ovf.

    Returns:
        NDArray[np.float32]: The generated pseudo-random array.

    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(nz, ny, nx, ncomp), dtype=np.float32)


def check_roundtrip(
    arr: NDArray[np.float32],
    tmp_path: Path,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dz: float = 1e-9,
) -> None:
    """Write, read back, and compare."""
    fname = tmp_path / "tmp.ovf"
    ovf.save_ovf(fname, arr, dx=dx, dy=dy, dz=dz)
    reloaded = ovf.load_ovf(fname)
    assert reloaded.dtype == np.float32  # OVF uses <f4
    assert reloaded.shape == arr.shape
    assert np.allclose(reloaded, arr, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 1),  # 1-component, thin film
        (2, 3, 4, 2),  # 2-component test
        (3, 4, 5, 3),  # full magnetisation vectors
    ],
)
def test_roundtrip_various_components(
    shape: tuple[int, int, int, int],
    tmp_path: Path,
) -> None:
    """Test OVF roundtrip for arrays with various numbers of components."""
    nz, ny, nx, ncomp = shape
    arr = make_array(nz, ny, nx, ncomp)
    check_roundtrip(arr, tmp_path)


@pytest.mark.parametrize(
    "spacing",
    [
        {"dx": 1e-9, "dy": 1e-9, "dz": 1e-9},
        {"dx": 2e-9, "dy": 1e-9, "dz": 0.5e-9},  # anisotropic voxel
    ],
)
def test_roundtrip_custom_spacing(spacing: dict[str, float], tmp_path: Path) -> None:
    """Test OVF roundtrip with custom voxel spacing."""
    arr = make_array(2, 2, 2, 3)
    check_roundtrip(arr, tmp_path, **spacing)


def test_get_ovf_parms(tmp_path: Path) -> None:
    """Test that get_ovf_parms returns correct parameters from a saved OVF file."""
    nx, ny, nz, ncomp = 4, 3, 2, 3
    dx, dy, dz = 2e-9, 3e-9, 4e-9
    arr = make_array(nz, ny, nx, ncomp)
    fname = tmp_path / "meta.ovf"
    ovf.save_ovf(fname, arr, dx=dx, dy=dy, dz=dz)

    p = ovf.get_ovf_parms(fname)
    assert p["Nx"] == nx
    assert p["Ny"] == ny
    assert p["Nz"] == nz
    assert math.isclose(p["dx"], dx, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["dy"], dy, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["dz"], dz, rel_tol=0, abs_tol=1e-15)
    assert p["comp"] == ncomp


def test_save_always_little_endian(tmp_path: Path) -> None:
    """Test that OVF files are always saved in little-endian format."""
    arr = make_array(1, 1, 4, 1)
    fname = tmp_path / "endian.ovf"
    ovf.save_ovf(fname, arr)
    with Path.open(fname, "rb") as f:
        while b"Begin: Data Binary 4" not in f.readline():
            pass
        (magic,) = np.fromfile(f, "<f4", count=1)
    assert math.isclose(magic, 1234567.0, rel_tol=0, abs_tol=0)
