from pathlib import Path
from typing import Literal

import numpy as np
import zarr
from zarr.storage import StoreLike
from zarr.core.group import AsyncGroup
from zarr.core.sync import sync

from .calc_modes import inner_calc_modes


class Pyzfn(zarr.Group):
    """
    A custom Zarr Group subclass for structured simulation output management.

    Provides utility methods and properties for handling hierarchical datasets,
    including array creation, metadata access, and visual tree formatting.
    """

    def __init__(self, store: StoreLike, **kwargs) -> None:
        """
        Initialize a Pyzfn group from a given Zarr store.

        Args:
            store (StoreLike): The Zarr store to back the group.
            **kwargs: Additional keyword arguments forwarded to AsyncGroup.
        """
        super().__init__(sync(AsyncGroup.from_store(store, overwrite=False, **kwargs)))
        self.clean_path: str = self.path.replace("file://", "")

    calc_modes = inner_calc_modes

    def __repr__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __str__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __name__(self) -> str:
        return self.name

    @property
    def path(self) -> str:
        """
        Full filesystem path of the Zarr store.

        Returns:
            str: Absolute file path with 'file://' prefix if available.
        """
        return str(self.store_path or Path("/tmp/test_only").absolute())

    @property
    def name(self) -> str:
        """
        Extract the base name of the Zarr store (without extension).

        Returns:
            str: The name of the group derived from the file path.
        """
        return self.path.split("/")[-1].replace(".zarr", "")

    @property
    def p(self) -> str:
        """
        Tree-like structure of the group and its contents (single-level).

        Returns:
            str: A string representation of the group tree.
        """
        return f"{self.name} {self.tree()}"

    def add_ndarray(
        self,
        name: str,
        data: np.ndarray,
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        overwrite: bool = True,
    ) -> zarr.Array:
        """
        Add a NumPy array to the Zarr group as a new dataset.

        Args:
            name (str): Name of the dataset to create.
            data (np.ndarray): NumPy array to store.
            chunks (tuple[int, ...] | Literal["auto"], optional): Chunk shape or "auto". Defaults to "auto".
            overwrite (bool, optional): Overwrite existing dataset if it exists. Defaults to True.

        Returns:
            zarr.Array: The created Zarr array.

        Raises:
            TypeError: If data is not a NumPy ndarray.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data)} instead.")

        dset = self.create_array(
            name=name,
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=overwrite,
            shards="auto",
        )
        dset[:] = data
        return dset
