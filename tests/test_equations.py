import math
from pyzfn import equations as eq


def test_kittel_default() -> None:
    """Default Kittel parameters should yield ≈ 3.00167 GHz."""
    val = eq.kittel_1948()
    assert math.isclose(val, 3.001673761e9, rel_tol=1e-6)


def test_kittel_0() -> None:
    """Default Kittel parameters should yield ≈ 3.00167 GHz."""
    assert eq.kittel_1948(Ku=1e9) == 0.0


def test_kalinikos_uniform_formula() -> None:
    B, Ms, Ku = 0.235, 1.15e6, 0.938e6
    val = eq.kalinikos_1986(B, Ms, Ku)
    gamma = eq._gamma(2.002)
    ref = gamma * (B - eq.MU0 * Ms + 2.0 * Ku / Ms) / (2.0 * math.pi)
    assert math.isclose(val, ref, rel_tol=1e-12)


def test_kalinikos_1986() -> None:
    val = eq.kalinikos_1986_no_approx(
        k=0.0,
        thickness=0.001,
        B=0.235,
        Ms=1.15e6,
        Aex=1.0e-11,
        Ku=0.938e6,
        g=2.002,
    )
    assert math.isclose(val, 24843610706.22996, rel_tol=1e-12)


def test_kim_dmi_sign() -> None:
    """Negative DMI lowers the +k Damon-Eshbach frequency."""
    f_neg = eq.kim_2016(1e7, dmi=-1e-5)
    f_pos = eq.kim_2016(1e7, dmi=+1e-5)
    assert f_neg < f_pos


def test_kim_wrong_disptype() -> None:
    """Wrong disptype should raise ValueError."""
    try:
        eq.kim_2016(1e7, disptype="wrong")
    except ValueError as e:
        assert str(e) == "disptype must be 'de' or 'bv'"
    else:
        assert False, "Expected ValueError not raised"


def test_nonnegative_everywhere() -> None:
    assert eq.kittel_1948(B=0.0) >= 0.0
    assert eq.bottcher_2021(k=0.0, B=0.0) >= 0.0


def test_cortes_ortuno_2013() -> None:
    """Test the Cortes-Ortuno 2013 equation."""
    k = 1e7
    Ms = 1.15e6
    Ku = 0.938e6
    Aex = 17e-12
    dmi = -1e-5
    B = 0.23
    thickness = 1.0e-9
    g = 2.002
    disptype = "de"

    freq = eq.cortes_ortuno_2013(k, Ms, Ku, Aex, dmi, B, thickness, g, disptype)
    assert math.isclose(freq, 2726541390.7433553, rel_tol=1e-12)


def test_cortes_ortuno_wrong_disptype() -> None:
    """Wrong disptype should raise ValueError."""
    try:
        eq.cortes_ortuno_2013(
            1e7, 1.15e6, 0.938e6, 17e-12, -1e-5, 0.23, 1.0e-9, 2.002, "wrong"
        )
    except ValueError as e:
        assert str(e) == "disptype must be 'de' or 'bv'"
    else:
        assert False, "Expected ValueError not raised"
