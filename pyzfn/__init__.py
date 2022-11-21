import os
from pathlib import Path
import shutil

import numpy as np
import zarr


def op(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path Not Found : '{path}'")
    if "ssh://" in path:
        return Group(zarr.storage.FSStore(path))
    else:
        return Group(zarr.storage.DirectoryStore(path))


class Group(zarr.hierarchy.Group):
    def __init__(self, store) -> None:
        zarr.hierarchy.Group.__init__(self, store)
        self.abs_path = Path(store.path).absolute()
        self.sim_name = self.abs_path.name.replace(self.abs_path.suffix, "")

    def __repr__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def __str__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def rm(self, dset: str):
        shutil.rmtree(f"{self.abs_path}/{dset}", ignore_errors=True)

    def mkdir(self, name: str):
        os.makedirs(f"{self.abs_path}/{name}", exist_ok=True)

    @property
    def pp(self):
        return self.tree(expand=True)

    @property
    def p(self):
        print(self.tree())
