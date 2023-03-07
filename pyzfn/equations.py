import numpy as np
from nptyping import Float32, NDArray, Shape

np1d = NDArray[Shape["*"], Float32]


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
    mu0 = 4 * np.pi * 1e-7
    lam_ex = 2 * Aex / (mu0 * Ms)
    Hu = (2 * Ku) / (mu0 * Ms)
    Hext = B / mu0  # - Ms
    eq1 = (gamma * mu0) / (2 * np.pi)
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
    mu0 = 4 * np.pi * 1e-7
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
                    + mu0 * Ms * thickness * kx**2 / np.abs(k) / 2
                )
                * (
                    B
                    + 2 * Aex * k**2 / Ms
                    - 2 * (Ku / Ms - mu0 * Nz * Ms / 2)
                    - mu0 * Ms * thickness * np.abs(k) / 2
                )
            )
            - 2 * gamma * dmi * kx / Ms
        )
    )
    return out


def kalinikos_k0(B: float, Ms: float, Ku: float, g: float) -> float:
    gamma = 87.9447e9 * g
    mu0 = np.pi * 4e-07
    H0 = B / mu0
    Hanis = 2 * Ku / Ms / mu0
    w0 = gamma * mu0 * (H0 - Ms + Hanis)
    w2 = w0**2
    out: float = np.sqrt(np.abs(w2)) / (2 * np.pi)
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
    mu0 = np.pi * 4e-07
    H0 = B / mu0
    wM = gamma * mu0 * Ms
    Hanis = 2 * Ku / Ms / mu0
    Hex = 2 * Aex / (mu0 * Ms) * k**2
    w0 = gamma * mu0 * (H0 - Ms + Hanis + Hex)
    w2 = w0 * (w0 + wM * (1 - (1 - np.exp(-k * thickness)) / (k * thickness)))
    out: np1d = np.sqrt(np.abs(w2)) / (2 * np.pi)
    return out


def kittel(
    B: float = 0.235, Ms: float = 1150e3, Ku: float = 938e3, g: float = 2.002
) -> float:
    gamma: float = 87.9447e9 * g
    mu0: float = 4 * np.pi * 1e-7
    H: float = B / mu0
    A: float = H * (H + Ms - 2 * Ku / mu0 / Ms)
    A = float(np.where(A < 0, 0, A))
    out: float = gamma * mu0 * np.sqrt(A) / (2 * np.pi)
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
    mu0 = 4 * np.pi * 1e-7
    H = B / mu0
    out: np1d = (1 / (2 * np.pi)) * (
        gamma
        * np.sqrt(
            (
                mu0 * (H + Ms * thickness * (np.cos(th) * k) ** 2 / 2 / np.abs(k))
                + 2 * Aex * k**2 / Ms
            )
            * (
                mu0 * (H + Ms * (1 - np.abs(k) * thickness / 2))
                + 2 * (Aex * k**2 - Ku) / Ms
            )
        )
        + 2 * gamma * dmi * np.cos(th) * k / Ms
    )
    return out
