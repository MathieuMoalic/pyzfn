import pytest
import zarr.storage
from pyzfn import Pyzfn
from collections.abc import Generator


@pytest.fixture
def base_sim() -> Generator[Pyzfn, None, None]:
    store = zarr.storage.MemoryStore()
    zarr.group(store=store, overwrite=True, zarr_format=2)
    sim = Pyzfn(store)
    yield sim
