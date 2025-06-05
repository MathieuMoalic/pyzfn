import colorsys
import os
import struct

import IPython
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotx
import numpy as np
import psutil

from pyzfn.chunks import calculate_largest_slice_points


def wisdom_name_from_array(arr):
    shape = "_".join([str(i) for i in arr.shape])
    return os.path.expanduser(f"~/.cache/fftw/{shape}_{arr.dtype}")


def make_cmap(min_color, max_color, mid_color=None, transparent_zero=False):
    cmap = np.ones((256, 4))
    for i in range(4):
        if mid_color is None:
            cmap[:, i] = np.linspace(min_color[i], max_color[i], 256) / 256
        else:
            cmap[:128, i] = np.linspace(min_color[i], mid_color[i], 128) / 256
            cmap[128:, i] = np.linspace(mid_color[i], max_color[i], 128) / 256
    if transparent_zero:
        cmap[128, 3] = 0
    return mcolors.ListedColormap(cmap)  # type: ignore


def save_ovf(path, arr, dx=1e-9, dy=1e-9, dz=1e-9):
    def whd(s):
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


def load_ovf(path):
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
        count = int(np.prod(dims) + 1)
        arr = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
    return arr


def get_ovf_parms(path):
    with open(path, "rb") as f:
        parms = {}
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


def get_slices(shape, chunks, slices):
    out = [[], [], [], []]
    for i, (s, c, sl) in enumerate(list(zip(shape, chunks, slices))[1:]):
        tmp_list = []
        for pt in list(range(s))[sl]:
            chunk_nb = pt // c
            if chunk_nb >= len(tmp_list):
                tmp_list.append([])
            tmp_list[chunk_nb].append(pt)
        for sublist in tmp_list:
            out[i].append(slice(min(sublist), max(sublist) + 1, sl.step))
    return out


def load_mpl_style(skip_style=False):
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
    if not skip_style:
        plt.style.use(matplotx.styles.dracula)
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["axes.grid.axis"] = "both"


def save_current_mplstyle(path):
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
        if len(v) > 4 and v[0] == "#":
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


def hsl2rgb(hsl):
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


def rgb2hsl(rgb):
    hsl = np.ones_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            r, g, b = rgb[i, j]
            h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
            hsl[i, j, 0] = h
            hsl[i, j, 1] = s
            hsl[i, j, 2] = l
    return hsl


def get_closest_point_on_fig(ptx, pty, linex, liney, fig):
    def normalize(point, line):
        line_norm = (line - line.min()) / (line.max() - line.min())
        point_norm = (point - line.min()) / (line.max() - line.min())
        return line_norm - point_norm

    figratio = fig.get_figwidth() / fig.get_figheight()
    ix = normalize(ptx, linex)
    iy = normalize(pty, liney) / figratio
    i = np.sqrt(ix**2 + iy**2).argmin()
    return int(i)


def indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = float(thres * (np.max(y) - np.min(y)) + np.min(y))

    min_dist = int(min_dist)
    dy = np.diff(y)

    zeros = np.where(dy == 0)[0]
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        zeros_diff = np.diff(zeros)
        zeros_diff_not_one = np.add(np.where(zeros_diff != 1), 1)[0]
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)
        for plateau in zero_plateaus:
            median = np.median(plateau)
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False
        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False
        peaks = np.arange(y.size)[~rem]  # type: ignore

    return peaks


def format_bytes(byte_size):
    if byte_size < 1024:
        return f"{byte_size} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"]
    size = float(byte_size)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def check_memory(slices, shape, force=False):
    largest_chunk_size = calculate_largest_slice_points(slices, shape) * 4
    available_memory = psutil.virtual_memory().available
    available_memory_str = format_bytes(available_memory)
    needed_memory = largest_chunk_size * 3
    needed_memory_str = format_bytes(needed_memory)
    if largest_chunk_size > needed_memory and not force:
        raise MemoryError(
            f"The needed memory to perform the fft ({needed_memory_str}) is larger than the available memory ({available_memory_str})."
            + " Please rechunk the data or pass `force=True` to the function to ignore this error."
        )
    return (
        f"Needed memory: {needed_memory_str} | Available memory: {available_memory_str}"
    )


def ellipticity(S, ground_state):
    if S.ndim != 4:
        raise ValueError("S must have 4 dimensions")
    if ground_state.ndim != 3:
        raise ValueError("ground_state must have 3 dimensions")
    if S.shape[1:] != ground_state.shape:
        raise ValueError("S and ground_state must have the same spatial dimensions")
    if S.shape[-1] != 3:
        raise ValueError("S must have 3 components in the last dimension")

    dot_vals = np.sum(S * ground_state, axis=-1)
    angles = np.arccos(np.clip(dot_vals, -1.0, 1.0))
    angle_min = angles.min(axis=0)
    angle_max = angles.max(axis=0)
    ellipticity = np.zeros_like(angle_min)
    zero_min = (angle_max != 0) & (angle_max != angle_min)
    ellipticity[zero_min] = angle_min[zero_min] / angle_max[zero_min]
    return ellipticity
