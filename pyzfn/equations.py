# flake8: noqa: N803, N806, PLR0913

"""Equations for ferromagnetic resonance and spin wave dispersion relations.

This module provides functions to compute resonance frequencies and dispersion
relations for various magnetic systems, including models by Kittel, Kalinikos & Slavin,
Böttcher, Kim, and Cortés-Ortuño. All equations assume SI units.
"""

from __future__ import annotations

import math

MU0: float = 4.0 * math.pi * 1.0e-7  # vacuum permeability  (N A⁻²)


def _gamma(g_factor: float) -> float:
    """Gyromagnetic ratio (rad s⁻¹ T⁻¹) from a Lande g-factor.

    Returns:
        Gyromagnetic ratio as a float (rad s⁻¹ T⁻¹).

    """
    return 87.9447e9 * g_factor


def kittel_1948(
    *,
    B: float,
    Ms: float,
    Ku: float,
    g: float,
) -> float:
    r"""Uniform in-plane ferromagnetic resonance (Kittel model).

    Calculates the *k* = 0 in-plane FMR frequency for a thin film with
    uniaxial perpendicular anisotropy, as derived by Kittel (1948).

    Parameters
    ----------
    B : float
        Static magnetic field in tesla (T), applied in the film plane.
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    g : float
        Landé *g*-factor (dimensionless).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz).

    Notes
    -----
    The frequency is given by

    .. math::

        f = \\frac{\\gamma}{2\\pi}
            \\sqrt{B\\left(B + \\mu_{0}M_{s} - 2K_{u}/M_{s}\\right)}.

    If the square-root argument becomes negative, the function returns
    ``0.0`` by definition.

    References
    ----------
    C. Kittel, *Phys. Rev.* **73**, 155 (1948).

    """
    r"""Uniform in-plane ferromagnetic resonance (Kittel 1948).

    Paper
        C. Kittel, *Phys. Rev.* **73**, 155 (1948)
        https://journals.aps.org/pr/abstract/10.1103/PhysRev.73.155

    Equation
        .. math::

           f \;=\;\frac{\gamma}{2\pi}
           \sqrt{B \bigl(B + \mu_{0}M_{s} - 2K_{u}/M_{s}\bigr)}.

    Limits / assumptions
        * Static field **B** and magnetisation in the film plane.
        * Only the lowest (k = 0) mode; uniaxial anisotropy axis ⟂ plane.
        * Returns **0.0 Hz** when the square-root argument is negative.

    Returns
    -------
        Resonance frequency in Hz as a float.

    """
    gamma = _gamma(g)
    radicand = B * (B + MU0 * Ms - 2.0 * Ku / Ms)
    return gamma * math.sqrt(radicand) / (2.0 * math.pi)


def kalinikos_1986(
    *,
    B: float,
    Ms: float,
    Ku: float,
    g: float,
) -> float:
    r"""Uniform perpendicular FMR (Kalinikos & Slavin 1986).

    Computes the *k* = 0 resonance frequency for a film magnetised
    **out of plane**.

    Parameters
    ----------
    B : float
        Static magnetic field in tesla (T), applied perpendicular to the film.
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    g : float
        Landé *g*-factor (dimensionless).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz).

    Notes
    -----
    The linear relation is

    .. math::

        f = \\frac{\\gamma}{2\\pi}\\bigl(B - \\mu_{0}M_{s} + 2K_{u}/M_{s}\\bigr).

    Valid only for fully saturated perpendicular magnetisation.

    References
    ----------
    B. A. Kalinikos & A. N. Slavin, *J. Phys. C* **19**, 7013 (1986).

    """
    gamma = _gamma(g)
    return gamma * (B - MU0 * Ms + 2.0 * Ku / Ms) / (2.0 * math.pi)


def kalinikos_1986_no_approx(
    *,
    k: float,
    thickness: float,
    B: float,
    Ms: float,
    Aex: float,
    Ku: float,
    g: float,
) -> float:
    r"""Full dipole-exchange dispersion (*n* = 0) after Kalinikos & Slavin.

    Evaluates the exact single-mode dispersion including exchange,
    dipolar and uniaxial-anisotropy contributions for arbitrary in-plane
    wavevector *k*.

    Parameters
    ----------
    k : float
        In-plane wavevector in reciprocal metres (m⁻¹).
    thickness : float
        Film thickness in metres (m).
    B : float
        Static magnetic field in tesla (T), applied perpendicular to the film.
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Aex : float
        Exchange stiffness in joules per metre (J m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    g : float
        Landé *g*-factor (dimensionless).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz). Returns ``0.0`` if the
        calculated radicand is negative.

    Notes
    -----
    * Uses the *n* = 0 dipolar form factor

      .. math:: F(k) = 1 - \\frac{1 - e^{-|k|t}}{|k|t}.

    * Neglects mode coupling (single-mode approximation).

    References
    ----------
    B. A. Kalinikos & A. N. Slavin, *J. Phys. C* **19**, 7013 (1986).

    """
    gamma = _gamma(g)
    H0 = B / MU0
    Han = 2.0 * Ku / (MU0 * Ms)
    Hex = 2.0 * Aex / (MU0 * Ms) * k * k
    w_M = gamma * MU0 * Ms
    w_0 = gamma * MU0 * (H0 - Ms + Han + Hex)

    Fk = (
        0.0
        if k * thickness == 0.0
        else (1.0 - math.exp(-abs(k) * thickness)) / (abs(k) * thickness)
    )
    w_sq = w_0 * (w_0 + w_M * (1.0 - Fk))
    return 0.0 if w_sq < 0.0 else math.sqrt(w_sq) / (2.0 * math.pi)


def bottcher_2021(
    *,
    k: float,
    Ms: float,
    Ku: float,
    Aex: float,
    B: float,
    thickness: float,
    g: float,
) -> float:
    r"""Perpendicular dipole-exchange spectrum (Böttcher *et al.* 2021).

    Calculates the dispersion for ultrathin films without interfacial DMI,
    valid for \\(|k t| \\lesssim 1\\).

    Parameters
    ----------
    k : float
        In-plane wavevector in reciprocal metres (m⁻¹).
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    Aex : float
        Exchange stiffness in joules per metre (J m⁻¹).
    B : float
        Static magnetic field in tesla (T), applied perpendicular to the film.
    thickness : float
        Film thickness in metres (m).
    g : float
        Landé *g*-factor (dimensionless).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz). Returns ``0.0`` when the
        radicand is negative.

    Notes
    -----
    The model keeps only the first exponential term of the demagnetising
    form factor and neglects interfacial DMI.

    References
    ----------
    T. Böttcher *et al.*, *IEEE Trans. Magn.* **57**, 9427561 (2021).

    """

    def _g(x: float) -> float:
        return 0.0 if x == 0.0 else 1.0 - (1.0 - math.exp(-abs(x))) / abs(x)

    gamma = _gamma(g)
    lam_ex = 2.0 * Aex / (MU0 * Ms)
    H_u = 2.0 * Ku / (MU0 * Ms)
    H_ext = B / MU0

    term2 = H_ext + lam_ex * k * k + Ms * _g(k * thickness)
    term3 = H_ext - H_u + lam_ex * k * k + Ms - Ms * _g(k * thickness)
    rad = term2 * term3
    pref = gamma * MU0 / (2.0 * math.pi)
    return 0.0 if rad < 0.0 else pref * math.sqrt(rad)


def kim_2016(
    *,
    k: float,
    Ms: float,
    Ku: float,
    Aex: float,
    dmi: float,
    B: float,
    thickness: float,
    g: float,
    disptype: str,
) -> float:
    """Thin-film spectra with interfacial DMI (Kim *et al.* 2016).

    Supports Damon-Eshbach (“de”) and backward-volume (“bv”) branches,
    including a linear ±*k* frequency shift from Néel-type DMI.

    Parameters
    ----------
    k : float
        In-plane wavevector in reciprocal metres (m⁻¹).
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    Aex : float
        Exchange stiffness in joules per metre (J m⁻¹).
    dmi : float
        Interfacial DMI constant in joules per square metre (J m⁻²).
    B : float
        Static magnetic field in tesla (T), applied perpendicular to the film.
    thickness : float
        Film thickness in metres (m).
    g : float
        Landé *g*-factor (dimensionless).
    disptype : {"de", "bv"}
        Dispersion type: Damon-Eshbach (“de”, θ = 0°) or backward-volume
        (“bv”, θ = 90°).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz).

    Raises
    ------
    ValueError
        If `disptype` is not ``"de"`` or ``"bv"``.

    Notes
    -----
    Ultrathin-film approximation: only first-order demagnetising
    corrections retained.

    References
    ----------
    J.-V. Kim *et al.*, *Phys. Rev. Lett.* **117**, 197204 (2016).

    """
    if disptype not in {"de", "bv"}:
        msg = "disptype must be 'de' or 'bv'"
        raise ValueError(msg)

    theta = 0.0 if disptype == "de" else math.pi / 2.0
    kx = k * -math.cos(theta)  # minus sign keeps legacy phase
    gamma = _gamma(g)

    exch = 2.0 * Aex * k * k / Ms
    dem_s = 0.0 if k == 0.0 else MU0 * Ms * thickness * kx * kx / (2.0 * abs(k))
    branch1 = B + exch + dem_s

    anis = 2.0 * (Ku / Ms - MU0 * Ms / 2.0)
    dem_b = 0.0 if k == 0.0 else MU0 * Ms * thickness * abs(k) / 2.0
    branch2 = B + exch - anis - dem_b

    pref = gamma / (2.0 * math.pi)
    freq = pref * math.sqrt(max(branch1 * branch2, 0.0))
    freq -= 2.0 * pref * dmi * kx / Ms  # DMI-induced ±k shift
    return freq


def cortes_ortuno_2013(
    *,
    k: float,
    Ms: float,
    Ku: float,
    Aex: float,
    dmi: float,
    B: float,
    thickness: float,
    g: float,
    disptype: str,
) -> float:
    """General thin-film dispersion with interfacial DMI.

    Implements the Cortés-Ortuño & Landeros (2013) model for both
    Damon-Eshbach and backward-volume geometries.

    Parameters
    ----------
    k : float
        In-plane wavevector in reciprocal metres (m⁻¹).
    Ms : float
        Saturation magnetisation in amperes per metre (A m⁻¹).
    Ku : float
        Uniaxial anisotropy constant in joules per cubic metre (J m⁻³).
    Aex : float
        Exchange stiffness in joules per metre (J m⁻¹).
    dmi : float
        Interfacial DMI constant in joules per square metre (J m⁻²).
    B : float
        Static magnetic field in tesla (T), applied perpendicular to the film.
    thickness : float
        Film thickness in metres (m).
    g : float
        Landé *g*-factor (dimensionless).
    disptype : {"de", "bv"}
        Dispersion type: Damon-Eshbach (“de”) or backward-volume (“bv”).

    Returns
    -------
    float
        Resonance frequency in hertz (Hz).

    Raises
    ------
    ValueError
        If `disptype` is not ``"de"`` or ``"bv"``.

    Notes
    -----
    Recovers the Kim 2016 result in the ultrathin (|kt| ≪ 1) limit and
    vanishing exchange.

    References
    ----------
    D. Cortés-Ortuño & P. Landeros, *J. Phys.: Condens. Matter* **25**, 156001 (2013).

    """
    if disptype not in {"de", "bv"}:
        msg = "disptype must be 'de' or 'bv'"
        raise ValueError(msg)

    theta = 0.0 if disptype == "de" else math.pi / 2.0
    gamma = _gamma(g)
    H = B / MU0

    demag = (
        0.0
        if k == 0.0
        else Ms * thickness * (math.cos(theta) * k) ** 2 / (2.0 * abs(k))
    )
    exch = 2.0 * Aex * k * k / Ms

    part1 = MU0 * (H + demag) + exch
    part2 = (
        MU0 * (H + Ms * (1.0 - abs(k) * thickness / 2.0))
        + 2.0 * (Aex * k * k - Ku) / Ms
    )

    pref = gamma / (2.0 * math.pi)
    freq = pref * math.sqrt(max(part1 * part2, 0.0))
    freq += 2.0 * pref * dmi * math.cos(theta) * k / Ms
    return freq
