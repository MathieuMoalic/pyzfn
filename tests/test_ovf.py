import math
import numpy as np
from pyzfn import ovf
import pytest
from typing import Any
from pathlib import Path
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_array(
    nz: int, ny: int, nx: int, ncomp: int, seed: int = 0
) -> NDArray[np.float32]:
    """Deterministic pseudo-random array in the exact dtype used by save_ovf."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(size=(nz, ny, nx, ncomp), dtype=np.float32)
    return data


def check_roundtrip(arr: NDArray[np.float32], tmp_path: Path, **kwargs: Any) -> None:
    """Write, read back, and compare."""
    fname = tmp_path / "tmp.ovf"
    ovf.save_ovf(fname, arr, **kwargs)
    reloaded = ovf.load_ovf(fname)
    assert reloaded.dtype == np.float32  # OVF uses <f4
    assert reloaded.shape == arr.shape
    assert np.allclose(reloaded, arr, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# parametrised round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 8, 1),  # 1-component, thin film
        (2, 3, 4, 2),  # 2-component test
        (3, 4, 5, 3),  # full magnetisation vectors
    ],
)
def test_roundtrip_various_components(
    shape: tuple[int, int, int, int], tmp_path: Path
) -> None:
    arr = make_array(*shape)
    check_roundtrip(arr, tmp_path)


@pytest.mark.parametrize(
    "spacing",
    [
        {"dx": 1e-9, "dy": 1e-9, "dz": 1e-9},
        {"dx": 2e-9, "dy": 1e-9, "dz": 0.5e-9},  # anisotropic voxel
    ],
)
def test_roundtrip_custom_spacing(spacing: dict[str, float], tmp_path: Path) -> None:
    arr = make_array(2, 2, 2, 3)
    check_roundtrip(arr, tmp_path, **spacing)


# ---------------------------------------------------------------------------
# metadata extraction
# ---------------------------------------------------------------------------


def test_get_ovf_parms(tmp_path: Path) -> None:
    nx, ny, nz, ncomp = 4, 3, 2, 3
    dx, dy, dz = 2e-9, 3e-9, 4e-9
    arr = make_array(nz, ny, nx, ncomp)
    fname = tmp_path / "meta.ovf"
    ovf.save_ovf(fname, arr, dx=dx, dy=dy, dz=dz)

    p = ovf.get_ovf_parms(fname)
    assert p["Nx"] == nx and p["Ny"] == ny and p["Nz"] == nz
    assert math.isclose(p["dx"], dx, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["dy"], dy, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(p["dz"], dz, rel_tol=0, abs_tol=1e-15)
    assert p["comp"] == ncomp


# ---------------------------------------------------------------------------
# dtype & alignment check
# ---------------------------------------------------------------------------


def test_save_always_little_endian(tmp_path: Path) -> None:
    arr = make_array(1, 1, 4, 1)
    fname = tmp_path / "endian.ovf"
    ovf.save_ovf(fname, arr)
    with open(fname, "rb") as f:
        while b"Begin: Data Binary 4" not in f.readline():
            pass
        (magic,) = np.fromfile(f, "<f4", count=1)
    assert math.isclose(magic, 1234567.0, rel_tol=0, abs_tol=0)
