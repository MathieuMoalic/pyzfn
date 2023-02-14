import numpy as np


def bottcher2021(
    k,
    Ms=1150e3,
    Ku=938e3,
    Aex=17e-12,
    B=0.23,
    dmi=0,
    thickness=1e-9,
    g=2.002,
    disptype="de",
):
    # https://ieeexplore.ieee.org/abstract/document/9427561
    def g_func(x):
        return 1 - (1 - np.exp(-np.abs(x))) / np.abs(x)

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
    k,
    Ms=1150e3,
    Ku=938e3,
    Aex=17e-12,
    dmi=-1e-5,
    B=0.23,
    thickness=1e-9,
    g=2.002,
    disptype="de",
):
    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.117.197204
    gamma = 87.9447e9 * g
    mu0 = 4 * np.pi * 1e-7
    Nz = 1
    # kx = k * np.sin(k_angle)
    kx = k * -np.cos({"de": 0, "bv": np.pi / 2}[disptype])
    return (
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


def kalinikos_k0(B, Ms, Ku, g):
    gamma = 87.9447e9 * g
    mu0 = np.pi * 4e-07
    H0 = B / mu0
    Hanis = 2 * Ku / Ms / mu0
    w0 = gamma * mu0 * (H0 - Ms + Hanis)
    w2 = w0**2
    return np.sqrt(np.abs(w2)) / (2 * np.pi)


def kalinikos(k, thickness, B, Ms, Aex, Ku, g=2.002):
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
    return np.sqrt(np.abs(w2)) / (2 * np.pi)


def kittel(B=0.235, Ms=1150e3, Ku=938e3, g=2.002):
    gamma = 87.9447e9 * g
    mu0 = 4 * np.pi * 1e-7
    H = B / mu0
    A = H * (H + Ms - 2 * Ku / mu0 / Ms)
    A = np.where(A < 0, 0, A)
    out = gamma * mu0 * np.sqrt(A) / (2 * np.pi)
    return out


def what_is_this(k, Ms, Ku, Aex, dmi, B, thickness, g, disptype):
    gamma = 87.9447e9 * g
    th = {"de": 0, "bv": np.pi / 2}[disptype]
    mu0 = 4 * np.pi * 1e-7
    H = B / mu0
    return (1 / (2 * np.pi)) * (
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
