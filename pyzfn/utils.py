import os
import pickle
import multiprocessing as mp
import struct
from typing import List, Optional, Tuple, Dict, Union

import matplotlib as mpl
import pyfftw
import numpy as np
from nptyping import NDArray, Float, Shape, Int

m_type = NDArray[Shape["*,*,*,*,*"], Float]


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
        print("Wisdom not found, it might take a while optimizing FFTW.")
        return False


def build_fft(
    arr: m_type, axis: Union[int, Tuple[int]], planner_effort: str = "FFTW_MEASURE"
) -> pyfftw.FFTW:
    wisdom_loaded = load_wisdom(arr)
    fft = pyfftw.builders.fft(
        arr,
        axis=axis,
        threads=mp.cpu_count() // 2,
        planner_effort=planner_effort,
        avoid_copy=True,
    )
    if not wisdom_loaded:
        save_wisdom(arr)
    return fft


def make_cmap(
    min_color: NDArray[Shape["256"], Float],
    max_color: NDArray[Shape["256"], Float],
    mid_color: Optional[NDArray[Shape["256"], Float]] = None,
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
