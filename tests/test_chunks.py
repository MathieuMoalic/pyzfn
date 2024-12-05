from typing import List, Tuple

import numpy as np
import pytest
import zarr

from pyzfn.chunks import calculate_largest_slice_points, get_zarr_chunk_slices


def create_zarr_array(shape: Tuple[int, ...], chunks: Tuple[int, ...]) -> zarr.Array:
    """
    Helper function to create a Zarr array with given shape and chunks.

    Parameters:
    ----------
    shape : Tuple[int, ...]
        Shape of the Zarr array.

    chunks : Tuple[int, ...]
        Chunk sizes for each dimension.

    Returns:
    -------
    zarr.Array
        The created Zarr array filled with random data.
    """
    store = zarr.MemoryStore()
    zarr_array = zarr.create(
        shape=shape,
        chunks=chunks,  # type: ignore
        dtype=np.float32,
        store=store,
        overwrite=True,
    )
    zarr_array[:] = np.random.random(shape)
    return zarr_array


@pytest.fixture
def mock_zarr_array() -> zarr.Array:
    """
    Fixture to create a mock Zarr array for testing.
    """
    # Example: 3D array
    shape = (10, 20, 30)
    chunks = (5, 20, 10)  # Dimension 1 is unchunked
    zarr_array = create_zarr_array(shape, chunks)
    return zarr_array


def test_simple1() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims: List[int] = []
    slices = [slice(None)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(0, 2),),
        (slice(2, 4),),
    ]
    assert slices_result == expected_slices


def test_simple2() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims = [0]
    slices = [slice(None)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(0, 4),),
    ]
    assert slices_result == expected_slices


def test_simple3() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims = [0]
    slices = [slice(1, 3)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(1, 3),),
    ]
    assert slices_result == expected_slices


def test_simple4() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims = [0]
    slices = [slice(0, 4)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(0, 4),),
    ]
    assert slices_result == expected_slices


def test_simple5() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims = [0]
    slices = [slice(0, 2)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(0, 2),),
    ]
    assert slices_result == expected_slices


def test_simple_error1() -> None:
    shape = (4,)
    chunks = (2,)
    no_chunk_dims = [1]
    slices = [slice(2, 4)]
    with pytest.raises(
        ValueError, match="Dimension 1 is out of bounds for array with 1 dimensions."
    ):
        get_zarr_chunk_slices(create_zarr_array(shape, chunks), no_chunk_dims, slices)


def test_simple_error2() -> None:
    shape = (0,)
    chunks = (2,)
    no_chunk_dims = [0]
    slices = [slice(2, 5)]
    with pytest.raises(ValueError, match="Dimension 0 has size 0."):
        get_zarr_chunk_slices(create_zarr_array(shape, chunks), no_chunk_dims, slices)


def test_2d_1() -> None:
    shape = (4, 8)
    chunks = (2, 8)
    no_chunk_dims = [0]
    slices = [slice(None), slice(None)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(0, 4), slice(0, 8)),
    ]
    assert slices_result == expected_slices


def test_2d_2() -> None:
    shape = (4, 8)
    chunks = (2, 8)
    no_chunk_dims: List[int] = []
    slices = [slice(1, 3), slice(None)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(1, 2), slice(0, 8)),
        (slice(2, 3), slice(0, 8)),
    ]
    assert slices_result == expected_slices


def test_no_overlap_in_slices() -> None:
    shape = (6, 8)
    chunks = (3, 4)
    no_chunk_dims: List[int] = []
    slices = [slice(10, 12), slice(5, 10)]  # No overlap in slices
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices: List[Tuple[slice, ...]] = []  # No slices should be generated
    assert slices_result == expected_slices


def test_single_element_case() -> None:
    shape = (6,)
    chunks = (2,)
    no_chunk_dims: List[int] = []
    slices = [slice(2, 3)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(2, 3),),
    ]
    assert slices_result == expected_slices


def test_multiple_no_chunk_dims() -> None:
    shape = (6, 8, 10)
    chunks = (3, 4, 5)
    no_chunk_dims = [0, 2]
    slices = [slice(1, 4), slice(2, 6), slice(4, 9)]
    slices_result = get_zarr_chunk_slices(
        create_zarr_array(shape, chunks), no_chunk_dims, slices
    )

    expected_slices = [
        (slice(1, 4), slice(2, 4), slice(4, 9)),
        (slice(1, 4), slice(4, 6), slice(4, 9)),
    ]
    assert slices_result == expected_slices


def test_single_slice_1d() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [(slice(0, 5),)]
    shape = (10,)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 5  # One slice covering the first 5 elements
    assert result == expected


def test_multiple_slices_1d() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [(slice(0, 5),), (slice(5, 10),)]
    shape = (10,)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 5  # The largest slice covers 5 elements
    assert result == expected


def test_multiple_slices_2d() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [
        (slice(0, 5), slice(0, 5)),
        (slice(5, 10), slice(0, 5)),
        (slice(0, 5), slice(5, 10)),
        (slice(5, 10), slice(5, 10)),
    ]
    shape = (10, 10)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 25  # The largest slice combination is 5x5 = 25 points
    assert result == expected


def test_multiple_slices_3d() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [
        (slice(0, 5), slice(0, 5), slice(0, 5)),
        (slice(5, 10), slice(0, 5), slice(0, 5)),
        (slice(0, 5), slice(5, 10), slice(0, 5)),
        (slice(0, 5), slice(0, 5), slice(5, 10)),
    ]
    shape = (10, 10, 10)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 125  # The largest slice combination is 5x5x5 = 125 points
    assert result == expected


def test_empty_slice_combinations() -> None:
    slice_combinations: List[Tuple[slice, ...]] = []
    shape = (10,)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 0  # No slices, so 0 points
    assert result == expected


def test_slice_with_step() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [(slice(0, 10, 2),)]
    shape = (10,)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 5  # Slice with step 2 covers every other element (0, 2, 4, 6, 8)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_single_point_slice() -> None:
    slice_combinations: List[Tuple[slice, ...]] = [(slice(5, 6),)]
    shape = (10,)
    result = calculate_largest_slice_points(slice_combinations, shape)
    expected = 1  # Single point slice (5,)
    assert result == expected, f"Expected {expected}, but got {result}"
