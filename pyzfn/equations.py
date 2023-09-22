from typing import Union
import numpy as np
from nptyping import Float32, NDArray, Shape

np1d = NDArray[Shape["*"], Float32]
optnp1d = Union[np1d, float, int]

MU0 = 4 * np.pi * 1e-7


def kittel(
    B: optnp1d = 0.235, Ms: optnp1d = 1150e3, Ku: optnp1d = 938e3, g: optnp1d = 2.002
) -> optnp1d:
    # this might be for inplane
    gamma: optnp1d = 87.9447e9 * g
    A: optnp1d = B * (B + Ms * MU0 - 2 * Ku / Ms)
    A = np.where(A < 0, 0, A)
    out: optnp1d = gamma * np.sqrt(A) / (2 * np.pi)
    return out


def kalinikos_k0(B: float, Ms: float, Ku: float, g: float) -> float:
    # this might be for out of plane
    gamma = 87.9447e9 * g
    Hanis = 2 * Ku / Ms
    out: float = gamma * (B - Ms * MU0 + Hanis) / (2 * np.pi)
    return out


def bottcher2021(
    k: np1d,
    Ms: float = 1150e3,
    Ku: float = 938e3,
    Aex: float = 17e-12,
    B: float = 0.23,
    thickness: float = 1e-9,
    g: float = 2.002,
) -> np1d:
    # https://ieeexplore.ieee.org/abstract/document/9427561
    def g_func(x: np1d) -> np1d:
        out: np1d = 1 - (1 - np.exp(-np.abs(x))) / np.abs(x)
        return out

    gamma = 87.9447e9 * g
    lam_ex = 2 * Aex / (MU0 * Ms)
    Hu = (2 * Ku) / (MU0 * Ms)
    Hext = B / MU0  # - Ms
    eq1 = (gamma * MU0) / (2 * np.pi)
    eq2 = Hext + lam_ex * k**2 + Ms * g_func(k * thickness)
    eq3 = Hext - Hu + lam_ex * k**2 + Ms - Ms * g_func(k * thickness)
    return eq1 * np.sqrt(eq2 * eq3)


def kim(
    k: np1d,
    Ms: float = 1150e3,
    Ku: float = 938e3,
    Aex: float = 17e-12,
    dmi: float = -1e-5,
    B: float = 0.23,
    thickness: float = 1e-9,
    g: float = 2.002,
    disptype: str = "de",
) -> np1d:
    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.117.197204
    gamma = 87.9447e9 * g
    Nz = 1
    kx = k * -np.cos({"de": 0, "bv": np.pi / 2}[disptype])
    out: np1d = (
        1
        / (2 * np.pi)
        * (
            gamma
            * np.sqrt(
                (
                    B
                    + 2 * Aex * k**2 / Ms
                    + MU0 * Ms * thickness * kx**2 / np.abs(k) / 2
                )
                * (
                    B
                    + 2 * Aex * k**2 / Ms
                    - 2 * (Ku / Ms - MU0 * Nz * Ms / 2)
                    - MU0 * Ms * thickness * np.abs(k) / 2
                )
            )
            - 2 * gamma * dmi * kx / Ms
        )
    )
    return out


def kalinikos(
    k: np1d,
    thickness: float,
    B: float,
    Ms: float,
    Aex: float,
    Ku: float,
    g: float = 2.002,
) -> np1d:
    # Something wrong
    print("This function needs fixing, frequencies are wrong")
    gamma = 87.9447e9 * g
    H0 = B / MU0
    wM = gamma * MU0 * Ms
    Hanis = 2 * Ku / Ms / MU0
    Hex = 2 * Aex / (MU0 * Ms) * k**2
    w0 = gamma * MU0 * (H0 - Ms + Hanis + Hex)
    w2 = w0 * (w0 + wM * (1 - (1 - np.exp(-k * thickness)) / (k * thickness)))
    out: np1d = np.sqrt(np.abs(w2)) / (2 * np.pi)
    return out


def what_is_this(
    k: np1d,
    Ms: float,
    Ku: float,
    Aex: float,
    dmi: float,
    B: float,
    thickness: float,
    g: float,
    disptype: str,
) -> np1d:
    gamma = 87.9447e9 * g
    th = {"de": 0, "bv": np.pi / 2}[disptype]
    H = B / MU0
    out: np1d = (1 / (2 * np.pi)) * (
        gamma
        * np.sqrt(
            (
                MU0 * (H + Ms * thickness * (np.cos(th) * k) ** 2 / 2 / np.abs(k))
                + 2 * Aex * k**2 / Ms
            )
            * (
                MU0 * (H + Ms * (1 - np.abs(k) * thickness / 2))
                + 2 * (Aex * k**2 - Ku) / Ms
            )
        )
        + 2 * gamma * dmi * np.cos(th) * k / Ms
    )
    return out
