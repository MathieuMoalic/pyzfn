import math
from pyzfn import equations as eq


def test_kittel_default() -> None:
    """Default Kittel parameters should yield ≈ 3.00167 GHz."""
    val = eq.kittel_1948()
    assert math.isclose(val, 3.001673761e9, rel_tol=1e-6)


def test_kalinikos_uniform_formula() -> None:
    B, Ms, Ku = 0.235, 1.15e6, 0.938e6
    val = eq.kalinikos_1986(B, Ms, Ku)
    gamma = eq._gamma(2.002)
    ref = gamma * (B - eq.MU0 * Ms + 2.0 * Ku / Ms) / (2.0 * math.pi)
    assert math.isclose(val, ref, rel_tol=1e-12)


def test_kim_dmi_sign() -> None:
    """Negative DMI lowers the +k Damon-Eshbach frequency."""
    f_neg = eq.kim_2016(1e7, dmi=-1e-5)
    f_pos = eq.kim_2016(1e7, dmi=+1e-5)
    assert f_neg < f_pos


def test_nonnegative_everywhere() -> None:
    assert eq.kittel_1948(B=0.0) >= 0.0
    assert eq.bottcher_2021(k=0.0, B=0.0) >= 0.0
