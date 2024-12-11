from pyzfn import Pyzfn as op


def test_Pyzfn() -> None:
    return


def test_calc_modes() -> None:
    job = op("/home/mat/z1/radial_vortex/other/old/1_test/v1.zarr")
    job.calc_modes()
