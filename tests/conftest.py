"""Pytest fixtures for pyzfn test suite.

Provides reusable fixtures for testing Pyzfn with in-memory Zarr stores.
"""

import pytest
import zarr.storage

from pyzfn import Pyzfn


@pytest.fixture
def base_sim() -> Pyzfn:
    """Fixture that provides a Pyzfn instance with an in-memory Zarr store.

    Returns:
        Pyzfn: An instance of Pyzfn using an in-memory Zarr store.

    """
    store = zarr.storage.MemoryStore()
    zarr.group(store=store, overwrite=True, zarr_format=2)
    return Pyzfn(store)
