import os
import pickle
import struct
import colorsys
from typing import List, Optional, Tuple, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
from ipympl.backend_nbagg import Canvas
import pyfftw
import numpy as np
import IPython
from nptyping import NDArray, Float, Shape, Int, Float32
import matplotx

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
) -> mpl.colors.ListedColormap:
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
    return mpl.colors.ListedColormap(cmap)


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


def hsl2rgb(hsl: NDArray[Shape["*,*,3"], Float32]) -> NDArray[Shape["*,*,3"], Float32]:
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]

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
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            hsl[i, j, 0] = h
            hsl[i, j, 1] = s
            hsl[i, j, 2] = l
    return hsl


def get_closest_point_on_fig(
    ptx: float, pty: float, linex: np1d, liney: np1d, fig: plt.Figure
) -> int:
    def normalize(point: float, line: np1d) -> np1d:
        line_norm = (line - line.min()) / (line.max() - line.min())
        point_norm = (point - line.min()) / (line.max() - line.min())
        out: np1d = line_norm - point_norm
        return out

    figratio = fig.get_figwidth() / fig.get_figheight()
    ix = normalize(ptx, linex)
    iy = normalize(pty, liney) / figratio
    i: int = np.sqrt(ix**2 + iy**2).argmin()
    return i
