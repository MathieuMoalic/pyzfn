import warnings
from pathlib import Path
from typing import Literal, TypeVar
from collections.abc import Sequence

import numpy as np
import zarr
from numpy.typing import NDArray
from rich.console import Console
from rich.tree import Tree
from zarr.core.group import AsyncGroup, Group
from zarr.core.sync import sync
from zarr.storage import StoreLike

from .calc_modes import inner_calc_modes
from .ispec import inner_ispec
from .snapshot import inner_snapshot

warnings.filterwarnings(
    "ignore",
    message="Object at .* is not recognized as a component of a Zarr hierarchy.",
    category=UserWarning,
)

T = TypeVar("T", bound=np.generic)
IndexLike = int | slice | Sequence[int] | NDArray[np.int_]
SliceTuple = tuple[IndexLike, ...]


class Pyzfn(Group):  # type: ignore[misc]
    """
    A custom Zarr Group subclass for structured simulation output management.

    Provides utility methods and properties for handling hierarchical datasets,
    including array creation, metadata access, and visual tree formatting.
    """

    def __init__(
        self,
        store: StoreLike,
        zarr_format: Literal[2, 3] = 2,
    ) -> None:
        """
        Initialize a Pyzfn group from a given Zarr store.

        Args:
            store (StoreLike): The Zarr store to back the group. Most commonly a string to a Zarr directory, i.e. "path/to/simulation.zarr".
            zarr_format (Literal[2, 3], optional): Zarr format version. Defaults to 3.
        """
        if isinstance(store, str) or isinstance(store, Path):
            p = Path(store)
            if not p.exists():
                raise FileNotFoundError(f"Path '{store}' does not exist.")
            if not p.is_dir():
                raise NotADirectoryError(f"Path '{store}' is not a directory.")
        super().__init__(sync(AsyncGroup.open(store, zarr_format=zarr_format)))
        self.clean_path: str = self.path.replace("file://", "")

    calc_modes = inner_calc_modes
    ispec = inner_ispec
    snapshot = inner_snapshot

    def __repr__(self) -> str:
        return f"Pyzfn('{self.name}')"

    def __str__(self) -> str:
        return f"Pyzfn('{self.name}')"

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
    def p(self) -> None:
        def add_to_tree(group, tree_node, prefix=""):  # type: ignore
            for key, node in sorted(group.members()):
                full_key = f"{prefix}/{key}" if prefix else key
                if hasattr(node, "members"):  # It's a group-like object
                    label = f"[bold]{key}[/bold]"
                    new_tree = tree_node.add(label)
                    add_to_tree(node, new_tree, full_key)
                else:
                    shape = getattr(node, "shape", "?")
                    dtype = getattr(node, "dtype", "?")
                    label = f"[bold]{key}[/bold] {shape} {dtype}"
                    tree_node.add(label)

        tree = Tree(f"[bold]{self.name}[/bold]")
        add_to_tree(self, tree)
        Console().print(tree)

    def rm(self, name: str) -> None:
        """
        Remove a dataset or group from the Zarr store.

        Args:
            name (str): Name of the dataset or group to remove.

        Raises:
            KeyError: If the specified name does not exist in the group.
        """
        if name not in self:
            raise KeyError(f"Dataset '{name}' not found in group '{self.name}'.")
        del self[name]

    def add_ndarray(
        self,
        name: str,
        data: NDArray[T],
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        overwrite: bool = True,
    ) -> zarr.Array:
        """
        Add a NumPy array to the Zarr group as a new dataset.

        Args:
            name (str): Name of the dataset to create.
            data (NDArray[T]): NumPy array to store.
            chunks (tuple[int, ...] | Literal["auto"], optional): Chunk shape or "auto". Defaults to "auto".
            overwrite (bool, optional): Overwrite existing dataset if it exists. Defaults to True.

        Returns:
            zarr.Array: The created Zarr array.

        Raises:
            TypeError: If data is not a NumPy ndarray.
        """
        dset = self.create_array(
            name=name,
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=overwrite,
            shards=None,
        )
        dset[:] = data
        return dset

    def get_mode(self, dset_str: str, f: float) -> NDArray[np.complex64]:
        """
        Retrieve a specific mode from the dataset based on frequency and spatial indices.
        Args:
            dset_str (str): Dataset name to retrieve modes from.
            f (float): Frequency to select the mode.
        Returns:
            NDArray[np.complex64]: The selected mode as a complex64 NumPy array.
        Raises:
            KeyError: If the dataset does not exist.
        """
        freqs = self.g(f"modes/{dset_str}/freqs")
        fi = int((np.abs(freqs - f)).argmin())
        return np.array(self.get_array(f"modes/{dset_str}/arr")[fi], dtype=np.complex64)

    def get_array(self, name: str) -> zarr.Array:
        """
        Retrieve a Zarr array by name.

        Args:
            name (str): Name of the dataset to retrieve.

        Returns:
            zarr.Array: The requested Zarr array.

        Raises:
            KeyError: If the dataset does not exist.
        """
        if name not in self:
            raise KeyError(f"Dataset '{name}' not found in group '{self.name}'.")
        array = self[name]
        if not isinstance(array, zarr.Array):
            raise ValueError("Array must be a zarr array not a Group")
        return array

    def g(
        self,
        dset_in_str: str,
        slices: SliceTuple | slice | None = None,
    ) -> NDArray[T]:
        """
        Retrieve a sliced view of a dataset as a NumPy array.

        Args:
            dset_in_str (str): Name of the dataset to retrieve from the Zarr group.
            slices (slice | tuple[slice, ...], optional): Slice or tuple of slices to apply.
                Defaults to a full slice of 5 dimensions (i.e., `slice(None)` * 5).
                Tip: use np.s_ to create complex slices.

        Returns:
            NDArray[T]: A NumPy array containing the sliced data, preserving the original dtype.

        Raises:
            KeyError: If the dataset does not exist.
            ValueError: If the dataset is not a Zarr array.
        """
        arr = self.get_array(dset_in_str)
        if slices is None:
            return np.asarray(arr[:])
        if isinstance(slices, slice):
            slices = (slices,)
        if len(slices) > arr.ndim:
            raise ValueError(
                f"Too many slices: got {len(slices)} for {arr.ndim}-dimensional array."
            )

        return np.asarray(arr[slices])  # type: ignore
