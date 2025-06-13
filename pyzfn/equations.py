from __future__ import annotations
import math

MU0: float = 4.0 * math.pi * 1.0e-7  # vacuum permeability  (N A⁻²)


def _gamma(g_factor: float) -> float:
    """Gyromagnetic ratio γ (rad s⁻¹ T⁻¹) from a Landé g-factor."""
    return 87.9447e9 * g_factor


def kittel_1948(
    B: float = 0.235,
    Ms: float = 1.150e6,
    Ku: float = 0.938e6,
    g: float = 2.002,
) -> float:
    r"""
    Uniform in-plane ferromagnetic resonance (Kittel 1948).

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
    """
    gamma = _gamma(g)
    radicand = B * (B + MU0 * Ms - 2.0 * Ku / Ms)
    if radicand < 0.0:
        return 0.0
    return gamma * math.sqrt(radicand) / (2.0 * math.pi)


def kalinikos_1986(
    B: float,
    Ms: float,
    Ku: float,
    g: float = 2.002,
) -> float:
    r"""
    Uniform (k = 0) perpendicular FMR - Kalinikos & Slavin 1986.

    Paper
        B. A. Kalinikos & A. N. Slavin, *J. Phys. C* **19**, 7013 (1986)
        https://iopscience.iop.org/article/10.1088/0022-3719/19/35/014/pdf

    .. math::

        f = \frac{\gamma}{2\pi} \bigl(B - \mu_{0} M_{s} + 2K_{u}/M_{s}\bigr).

    Valid only when the magnetisation is saturated out of the film plane.
    """
    gamma = _gamma(g)
    return gamma * (B - MU0 * Ms + 2.0 * Ku / Ms) / (2.0 * math.pi)


def kalinikos_1986_no_approx(
    k: float,
    thickness: float,
    B: float,
    Ms: float,
    Aex: float,
    Ku: float,
    g: float = 2.002,
) -> float:
    r"""
    Full dipole-exchange dispersion, n = 0 (Kalinikos & Slavin 1986).

    Paper
        B. A. Kalinikos & A. N. Slavin, *J. Phys. C* **19**, 7013 (1986)
        https://iopscience.iop.org/article/10.1088/0022-3719/19/35/014/pdf

    Notes
        * Uses the exact n = 0 dipolar form-factor
          :math:`F(k) = 1 - \tfrac{1 - e^{-|k|t}}{|k|t}`.
        * Single-mode approximation (ignores mode coupling).
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
    k: float,
    Ms: float = 1.150e6,
    Ku: float = 0.938e6,
    Aex: float = 17e-12,
    B: float = 0.23,
    thickness: float = 1.0e-9,
    g: float = 2.002,
) -> float:
    r"""
    Perpendicular dipole-exchange dispersion (Böttcher 2021).

    Paper
        T. Böttcher *et al.*, *IEEE Trans. Magn.* **57**, 9427561 (2021)
        https://ieeexplore.ieee.org/abstract/document/9427561

    Assumptions
        * Magnetisation saturated out-of-plane.
        * Interfacial DMI **neglected**.
        * Valid while |k·t| ≲ 1 (uses first exponential form-factor).
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
    k: float,
    Ms: float = 1.150e6,
    Ku: float = 0.938e6,
    Aex: float = 17e-12,
    dmi: float = -1e-5,
    B: float = 0.23,
    thickness: float = 1.0e-9,
    g: float = 2.002,
    disptype: str = "de",
) -> float:
    r"""
    Surface (DE) or bulk-volume (BV) modes with interfacial DMI.

    Paper
        J.-V. Kim *et al.*, *Phys. Rev. Lett.* **117**, 197204 (2016)
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.117.197204

    Parameters
        disptype - ``"de"`` (Damon-Eshbach, θ = 0°) or ``"bv"`` (θ = 90°).

    Assumptions
        * Ultrathin limit: only first demagnetising corrections kept.
        * Linear ±k shift from interfacial (Néel) DMI.
    """
    if disptype not in {"de", "bv"}:
        raise ValueError("disptype must be 'de' or 'bv'")

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
    r"""
    General thin-film spectrum with interfacial DMI (Cortés-Ortuño 2013).

    Paper
        D. Cortés-Ortuño & P. Landeros, *J. Phys.: Condens. Matter* **25**, 156001 (2013)
        https://iopscience.iop.org/article/10.1088/0953-8984/25/15/156001/pdf
    """
    if disptype not in {"de", "bv"}:
        raise ValueError("disptype must be 'de' or 'bv'")

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
