from pyzfn import Pyzfn
import zarr

import pytest


def test_Pyzfn() -> None:
    with pytest.raises(Exception):
        Pyzfn("does_not_exist")
    Pyzfn("tests/test.zarr")


def test_getitem() -> None:
    job = Pyzfn("tests/test.zarr")
    job["m"]


def test_getattr() -> None:
    job = Pyzfn("tests/test.zarr")
    assert isinstance(job.m, zarr.Array)
    assert isinstance(job.dx, float)
    assert isinstance(job.Nx, int)


def test_repr() -> None:
    job = Pyzfn("tests/test.zarr")
    assert str(job) == "Pyzfn('test')"


def test_p() -> None:
    job = Pyzfn("tests/test.zarr")
    job.p


def test_calc_disp() -> None:
    job = Pyzfn("tests/test.zarr")
    job.calc_disp()
    job.rm("disp")


def test_calc_modes() -> None:
    job = Pyzfn("tests/test.zarr")
    job.calc_modes()
    job.rm("modes")
    job.rm("fft")


def test_snapshot() -> None:
    job = Pyzfn("tests/test.zarr")
    job.snapshot()
