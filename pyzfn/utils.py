import colorsys
import os
import pickle
import struct
from typing import Any, Dict, List, Optional, Tuple

import IPython
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotx
import numpy as np
import pyfftw
from ipympl.backend_nbagg import Canvas
from matplotlib.figure import Figure
from nptyping import Float, Float32, Int, NDArray, Shape

m_type = NDArray[Shape["*,*,*,*,*"], Float32]

np1d = NDArray[Shape["*"], Float32]
np2d = NDArray[Shape["*,*"], Float32]
np3d = NDArray[Shape["*,*,*"], Float32]
np4d = NDArray[Shape["*,*,*,*"], Float32]
np5d = NDArray[Shape["*,*,*,*,*"], Float32]


def wisdom_name_from_array(arr: m_type) -> str:
    shape = "_".join([str(i) for i in arr.shape])
    return os.path.expanduser(f"~/.cache/fftw/{shape}_{arr.dtype}")


def save_wisdom(arr: m_type) -> None:
    os.makedirs(os.path.expanduser("~/.cache/fftw/"), exist_ok=True)
    with open(wisdom_name_from_array(arr), "wb") as f:
        pickle.dump(pyfftw.export_wisdom(), f)


def load_wisdom(arr: m_type) -> bool:
    wisdom_path = wisdom_name_from_array(arr)
    if os.path.exists(wisdom_path):
        with open(wisdom_path, "rb") as f:
            pyfftw.import_wisdom(pickle.load(f))
        # print("Wisdom found.")
        return True
    else:
        # print("Wisdom not found, it might take a while optimizing FFTW.")
        return False


def make_cmap(
    min_color: Tuple[int, int, int, int],
    max_color: Tuple[int, int, int, int],
    mid_color: Optional[Tuple[int, int, int, int]] = None,
    transparent_zero: bool = False,
) -> mcolors.ListedColormap:
    cmap: NDArray[Shape["256, 4"], Float] = np.ones((256, 4))
    for i in range(4):
        if mid_color is None:
            cmap[:, i] = np.linspace(min_color[i], max_color[i], 256) / 256
        else:
            cmap[: 256 // 2, i] = (
                np.linspace(min_color[i], mid_color[i], 256 // 2) / 256
            )
            cmap[256 // 2 :, i] = (
                np.linspace(mid_color[i], max_color[i], 256 // 2) / 256
            )
    if transparent_zero:
        cmap[256 // 2, 3] = 0
    return mcolors.ListedColormap(cmap)


def save_ovf(
    path: str, arr: m_type, dx: float = 1e-9, dy: float = 1e-9, dz: float = 1e-9
) -> None:
    """Saves the given dataset for a given t to a valid OOMMF V2 ovf file"""

    def whd(s: str) -> None:
        s += "\n"
        f.write(s.encode("ASCII"))

    out = arr.astype("<f4").tobytes()

    xnodes, ynodes, znodes = arr.shape[2], arr.shape[1], arr.shape[0]
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = xnodes * dx, ynodes * dy, znodes * dz
    xbase, ybase, _ = dx / 2, dy / 2, dz / 2
    valuedim = arr.shape[-1]
    valuelabels = "x y z"
    valueunits = "1 1 1"
    total_sim_time = "0"
    name = path.split("/")[-1]
    with open(path, "wb") as f:
        whd("# OOMMF OVF 2.0")
        whd("# Segment count: 1")
        whd("# Begin: Segment")
        whd("# Begin: Header")
        whd(f"# Title: {name}")
        whd("# meshtype: rectangular")
        whd("# meshunit: m")
        whd(f"# xmin: {xmin}")
        whd(f"# ymin: {ymin}")
        whd(f"# zmin: {zmin}")
        whd(f"# xmax: {xmax}")
        whd(f"# ymax: {ymax}")
        whd(f"# zmax: {zmax}")
        whd(f"# valuedim: {valuedim}")
        whd(f"# valuelabels: {valuelabels}")
        whd(f"# valueunits: {valueunits}")
        whd(f"# Desc: Total simulation time:  {total_sim_time}  s")
        whd(f"# xbase: {xbase}")
        whd(f"# ybase: {ybase}")
        whd(f"# zbase: {ybase}")
        whd(f"# xnodes: {xnodes}")
        whd(f"# ynodes: {ynodes}")
        whd(f"# znodes: {znodes}")
        whd(f"# xstepsize: {dx}")
        whd(f"# ystepsize: {dy}")
        whd(f"# zstepsize: {dz}")
        whd("# End: Header")
        whd("# Begin: Data Binary 4")
        f.write(struct.pack("<f", 1234567.0))
        f.write(out)
        whd("# End: Data Binary 4")
        whd("# End: Segment")


def load_ovf(path: str) -> NDArray[Shape["4"], Int]:
    with open(path, "rb") as f:
        dims = np.array([0, 0, 0, 0])
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                dims[3] = int(line.split(" ")[-1])
            if "xnodes" in line:
                dims[2] = int(line.split(" ")[-1])
            if "ynodes" in line:
                dims[1] = int(line.split(" ")[-1])
            if "znodes" in line:
                dims[0] = int(line.split(" ")[-1])
            if "Begin: Data" in line:
                break
        count = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
        arr = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
    return arr


def get_ovf_parms(path: str) -> Dict[str, float]:
    with open(path, "rb") as f:
        parms: Dict[str, float] = {}
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                parms["comp"] = int(line.split(" ")[-1], 10)
            if "xnodes" in line:
                parms["Nx"] = int(line.split(" ")[-1])
            if "ynodes" in line:
                parms["Ny"] = int(line.split(" ")[-1])
            if "znodes" in line:
                parms["Nz"] = int(line.split(" ")[-1])
            if "xstepsize" in line:
                parms["dx"] = float(line.split(" ")[-1])
            if "ystepsize" in line:
                parms["dy"] = float(line.split(" ")[-1])
            if "zstepsize" in line:
                parms["dz"] = float(line.split(" ")[-1])
            if "Begin: Data" in line:
                break
    return parms


def get_slices(
    shape: Tuple[int, int, int, int, int],
    chunks: Tuple[int, int, int, int, int],
    slices: Tuple[slice, slice, slice, slice, slice],
) -> List[List[slice]]:
    out: List[List[slice]] = [[], [], [], []]
    for i, (s, c, sl) in enumerate(list(zip(shape, chunks, slices))[1:]):
        tmp_list: List[List[int]] = []
        for pt in list(range(s))[sl]:
            chunk_nb = pt // c
            if chunk_nb >= len(tmp_list):
                tmp_list.append([])
            tmp_list[chunk_nb].append(pt)
        for sublist in tmp_list:
            out[i].append(slice(min(sublist), max(sublist) + 1, sl.step))
    return out


def load_mpl_style(skip_style: bool = False) -> None:
    ipy = IPython.get_ipython()  # type: ignore
    if ipy is not None:
        ipy.run_cell_magic(
            "html",
            "",
            """<style>
                .cell-output-ipywidget-background {
                    background-color: transparent !important;
                }
                .jupyter-matplotlib-footer {
                    color: white !important;
                }
                :root {
                    --jp-widgets-color: var(--vscode-editor-foreground);
                    --jp-widgets-font-size: var(--vscode-editor-font-size);
                }  
               </style>""",
        )
        ipy.run_line_magic("matplotlib", "widget")
        ipy.run_line_magic("load_ext", "autoreload")
        ipy.run_line_magic("autoreload", "2")

        plt.rcParams["figure.max_open_warning"] = 1000

        Canvas.header_visible.default_value = False
        Canvas.footer_visible.default_value = True
    if not skip_style:
        plt.style.use(matplotx.styles.dracula)
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.rcParams["figure.autolayout"] = True
        # plt.rcParams["axes.grid"] = True
        plt.rcParams["axes.grid.axis"] = "both"


def save_current_mplstyle(path: str) -> None:
    newlines = []
    for k, v in plt.rcParams.items():
        if k in [
            "backend",
            "backend_fallback",
            "date.epoch",
            "docstring.hardcopy",
            "figure.max_open_warning",
            "figure.raise_window",
            "interactive",
            "savefig.directory",
            "timezone",
            "tk.window_focus",
            "toolbar",
            "webagg.address",
            "webagg.open_in_browser",
            "webagg.port",
            "webagg.port_retries",
        ]:
            continue
        if isinstance(v, list):
            v = [str(i) for i in v]
            v = ", ".join(v)
        v = str(v)
        if len(v) > 4:
            if v[0] == "#":
                v = v[1:]
        if k == "axes.prop_cycle":
            v = v.replace("#", "")
        if k == "grid.color":
            v = '"' + v + '"'
        if k == "lines.dash_joinstyle":
            v = "round"
        if k == "lines.dash_capstyle":
            v = "butt"
        if k == "lines.solid_capstyle":
            v = "projecting"
        if k == "lines.solid_joinstyle":
            v = "round"
        if k == "savefig.bbox":
            v = "tight"
        newlines.append(k + ": " + v + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(newlines)


def hsl2rgb(hsl: np3d) -> np3d:
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]  # noqa: E741

    rgb = np.zeros_like(hsl)
    for i, n in enumerate([0, 8, 4]):
        k = (n + h / 30) % 12
        a = s * np.minimum(l, 1 - l)
        k = np.minimum(k - 3, 9 - k)
        k = np.clip(k, -1, 1)
        rgb[..., i] = l - a * k
    rgb = np.clip(rgb, 0, 1)
    return rgb


def rgb2hsl(rgb: np3d) -> np3d:
    hsl = np.ones_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            r, g, b = rgb[i, j]
            h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
            hsl[i, j, 0] = h
            hsl[i, j, 1] = s
            hsl[i, j, 2] = l
    return hsl


def get_closest_point_on_fig(
    ptx: float, pty: float, linex: np1d, liney: np1d, fig: Figure
) -> int:
    def normalize(point: float, line: np1d) -> np1d:
        line_norm = (line - line.min()) / (line.max() - line.min())
        point_norm = (point - line.min()) / (line.max() - line.min())
        out: np1d = line_norm - point_norm
        return out

    figratio = fig.get_figwidth() / fig.get_figheight()
    ix = normalize(ptx, linex)
    iy = normalize(pty, liney) / figratio
    i: np.intp = np.sqrt(ix**2 + iy**2).argmin()
    return int(i)


def indexes(
    y: np1d, thres: float = 0.3, min_dist: int = 1, thres_abs: bool = False
) -> NDArray[Any, Any]:
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value, instead of
        a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected.
        When using with Pandas DataFrames, iloc should be used to access the values at the returned positions.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = float(thres * (np.max(y) - np.min(y)) + np.min(y))

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    (zeros,) = np.where(dy == 0)
    (zeros,) = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        (zeros_diff_not_one,) = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks
