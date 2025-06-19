"""Unit tests for the equations module in pyzfn.

This module tests various magnetic resonance and spin wave frequency equations
implemented in pyzfn.equations.
"""

import math

import pytest

from pyzfn import equations as eq


def test_kittel_default() -> None:
    """Default Kittel parameters should yield ≈ 3.00167 GHz."""
    val = eq.kittel_1948(
        B=0.235,
        Ms=1.15e6,
        Ku=0.938e6,
        g=2.002,
    )
    assert math.isclose(val, 3.001673761e9, rel_tol=1e-6)


def test_kittel_0() -> None:
    """Default Kittel parameters should yield ≈ 3.00167 GHz."""
    assert (
        eq.kittel_1948(
            B=0.0,
            Ms=1.15e6,
            Ku=1e9,  # Arbitrary large value
            g=2.002,
        )
        == 0.0
    )


def test_kalinikos_uniform_formula() -> None:
    """Test the uniform mode Kalinikos 1986 formula against a reference calculation."""
    val = eq.kalinikos_1986(
        B=0.235,
        Ms=1.15e6,
        Ku=0.938e6,
        g=2.002,
    )
    assert math.isclose(val, 11801931410.075216, rel_tol=1e-12)


def test_kalinikos_1986_no_approx() -> None:
    """Test the Kalinikos 1986 no-approximation formula against a reference value."""
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
    f_neg = eq.kim_2016(
        k=1e7,
        Ms=1.15e6,
        Ku=0.938e6,
        Aex=17e-12,
        dmi=-1e-5,
        B=0.23,
        thickness=1.0e-9,
        g=2.002,
        disptype="de",
    )
    f_pos = eq.kim_2016(
        k=1e7,
        Ms=1.15e6,
        Ku=0.938e6,
        Aex=17e-12,
        dmi=+1e-5,
        B=0.23,
        thickness=1.0e-9,
        g=2.002,
        disptype="de",
    )
    assert f_neg < f_pos


def test_kim_wrong_disptype() -> None:
    """Wrong disptype should raise ValueError."""
    with pytest.raises(ValueError, match="disptype must be 'de' or 'bv'"):
        eq.kim_2016(
            k=1e7,
            Ms=1.15e6,
            Ku=0.938e6,
            Aex=17e-12,
            dmi=-1e-5,
            B=0.23,
            thickness=1.0e-9,
            g=2.002,
            disptype="wrong",  # Invalid disptype
        )


def test_nonnegative_everywhere() -> None:
    """Test that all relevant equations return nonnegative values for zero field."""
    assert (
        eq.kittel_1948(
            B=0.0,
            Ms=1.15e6,
            Ku=0.938e6,
            g=2.002,
        )
        >= 0.0
    )
    assert (
        eq.bottcher_2021(
            k=0.0,
            B=0.0,
            Ms=1.15e6,
            Ku=0.938e6,
            Aex=17e-12,
            thickness=1.0e-9,
            g=2.002,
        )
        >= 0.0
    )


def test_cortes_ortuno_2013() -> None:
    """Test the Cortes-Ortuno 2013 equation."""
    freq = eq.cortes_ortuno_2013(
        k=1e7,
        Ms=1.15e6,
        Ku=0.938e6,
        Aex=17e-12,
        dmi=-1e-5,
        B=0.23,
        thickness=1.0e-9,
        g=2.002,
        disptype="de",
    )
    assert math.isclose(freq, 2726541390.7433553, rel_tol=1e-12)


def test_cortes_ortuno_wrong_disptype() -> None:
    """Wrong disptype should raise ValueError."""
    with pytest.raises(ValueError, match="disptype must be 'de' or 'bv'"):
        eq.cortes_ortuno_2013(
            k=1e7,
            Ms=1.15e6,
            Ku=0.938e6,
            Aex=17e-12,
            dmi=-1e-5,
            B=0.23,
            thickness=1.0e-9,
            g=2.002,
            disptype="wrong",  # Invalid disptype
        )
