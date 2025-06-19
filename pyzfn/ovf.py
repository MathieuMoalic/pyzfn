# ruff: noqa: PLR0913
"""Read, write, and inspect OOMMF OVF 2.0 files."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from numpy.typing import NDArray

_ASCII: Final = "ASCII"
_MAGIC: Final[float] = 1_234_567.0


def _assemble_header(
    *,
    title: str,
    xnodes: int,
    ynodes: int,
    znodes: int,
    dx: float,
    dy: float,
    dz: float,
    valuedim: int,
) -> bytes:
    """Return a ready-to-write byte-string with the OVF header.

    Returns:
        bytes: The OVF header as a byte string, ready to be written to a file.

    """
    xmax, ymax, zmax = xnodes * dx, ynodes * dy, znodes * dz
    xbase, ybase, zbase = dx / 2, dy / 2, dz / 2

    header = [
        "# OOMMF OVF 2.0",
        "# Segment count: 1",
        "# Begin: Segment",
        "# Begin: Header",
        f"# Title: {title}",
        "# meshtype: rectangular",
        "# meshunit: m",
        "# xmin: 0",
        "# ymin: 0",
        "# zmin: 0",
        f"# xmax: {xmax}",
        f"# ymax: {ymax}",
        f"# zmax: {zmax}",
        f"# valuedim: {valuedim}",
        "# valuelabels: x y z",
        "# valueunits: 1 1 1",
        "# Desc: Total simulation time:  0  s",
        f"# xbase: {xbase}",
        f"# ybase: {ybase}",
        f"# zbase: {zbase}",
        f"# xnodes: {xnodes}",
        f"# ynodes: {ynodes}",
        f"# znodes: {znodes}",
        f"# xstepsize: {dx}",
        f"# ystepsize: {dy}",
        f"# zstepsize: {dz}",
        "# End: Header",
        "# Begin: Data Binary 4",
    ]
    # Join once and add final newline so later binary write starts on new line.
    return ("\n".join(header) + "\n").encode(_ASCII)


def save_ovf(
    path: str | Path,
    array: NDArray[np.float32],
    *,
    dx: float = 1e-9,
    dy: float = 1e-9,
    dz: float = 1e-9,
) -> None:
    """Write *array* to an OVF 2.0 file at *path*.

    The array shape must be ``(Nz, Ny, Nx, valuedim)``.
    Cell sizes (in metres) may be overridden via *dx*, *dy*, *dz*.
    """
    path = Path(path)
    nz, ny, nx, comps = array.shape
    header = _assemble_header(
        title=path.name,
        xnodes=nx,
        ynodes=ny,
        znodes=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        valuedim=comps,
    )

    with path.open("wb") as fp:
        fp.write(header)
        fp.write(struct.pack("<f", _MAGIC))
        fp.write(array.astype("<f4").tobytes())
        fp.write(b"\n# End: Data Binary 4\n# End: Segment\n")


def load_ovf(path: str | Path) -> NDArray[np.float32]:
    """Return the data contained in an OVF 2.0 file.

    The returned array has shape ``(Nz, Ny, Nx, valuedim)`` and
    dtype ``np.float32``.

    Returns:
        NDArray[np.float32]: The data array with shape (Nz, Ny, Nx, valuedim).

    """
    path = Path(path)
    with path.open("rb") as fp:
        dims = np.zeros(4, dtype=np.int32)  # [Nz, Ny, Nx, valuedim]
        for raw in fp:
            line = raw.strip().decode(_ASCII)
            if line.startswith("# xnodes:"):
                dims[2] = int(line.split()[-1])
            elif line.startswith("# ynodes:"):
                dims[1] = int(line.split()[-1])
            elif line.startswith("# znodes:"):
                dims[0] = int(line.split()[-1])
            elif line.startswith("# valuedim:"):
                dims[3] = int(line.split()[-1])
            elif "Begin: Data" in line:
                break

        count = int(dims.prod()) + 1  # +1 for the magic float
        return np.fromfile(fp, "<f4", count=count, sep="")[1:].reshape(dims)


def get_ovf_parms(path: str | Path) -> Mapping[str, int | float]:
    """Return key grid parameters from an OVF 2.0 header.

    Returns:
        Mapping[str, int | float]: Dictionary containing grid parameters such as
        'Nx', 'Ny', 'Nz', 'comp', 'dx', 'dy', and 'dz'.

    """
    keys: dict[str, int | float] = {}
    with Path(path).open("rb") as fp:
        for raw in fp:
            line = raw.strip().decode(_ASCII)
            if line.startswith("# xnodes:"):
                keys["Nx"] = int(line.split()[-1])
            elif line.startswith("# ynodes:"):
                keys["Ny"] = int(line.split()[-1])
            elif line.startswith("# znodes:"):
                keys["Nz"] = int(line.split()[-1])
            elif line.startswith("# valuedim:"):
                keys["comp"] = int(line.split()[-1])
            elif line.startswith("# xstepsize:"):
                keys["dx"] = float(line.split()[-1])
            elif line.startswith("# ystepsize:"):
                keys["dy"] = float(line.split()[-1])
            elif line.startswith("# zstepsize:"):
                keys["dz"] = float(line.split()[-1])
            elif "Begin: Data" in line:
                break
    return keys
