from pyzfn import Pyzfn

import pytest


def test_calc_modes() -> None:
    with pytest.raises(Exception):
        Pyzfn("dwa")
