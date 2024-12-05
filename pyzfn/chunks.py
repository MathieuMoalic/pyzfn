import itertools
from typing import List, Sequence, Tuple

import zarr


def get_zarr_chunk_slices(
    zarr_array: zarr.Array, no_chunk_dims: Sequence[int], slices: Sequence[slice]
) -> List[Tuple[slice, ...]]:
    """
    Generate a list of slice tuples representing the chunks of a Zarr array,
    ensuring that specified dimensions are not chunked (i.e., the entire dimension is included at once),
    and applying additional slicing constraints.

    Parameters:
    ----------
    zarr_array : zarr.core.Array
        The input Zarr array.

    no_chunk_dims : List[int]
        A list of dimension indices (0-based) that should not be chunked.
        The entire dimension will be included in each slice, regardless of chunking.

    slices : List[slice]
        A list of slice objects (one per dimension) to apply to each chunk slice.

    Returns:
    -------
    List[Tuple[slice, ...]]
        A list where each element is a tuple of slice objects corresponding to a chunk,
        filtered by the provided slices.

    Raises:
    ------
    ValueError:
        If any dimension in `no_chunk_dims` is out of bounds,
        if any dimension's size or chunk size is zero,
        or if the lengths of `no_chunk_dims` and `slices` do not match the number of dimensions.

    TypeError:
        If any element in `slices` is not a slice object.
    """
    ndim = zarr_array.ndim
    shape = zarr_array.shape
    chunks = zarr_array.chunks

    # Validate lengths of no_chunk_dims and slices
    # if len(no_chunk_dims) != ndim:
    #     raise ValueError(
    #         f"Length of no_chunk_dims ({len(no_chunk_dims)}) does not match number of dimensions ({ndim})."
    #     )
    if len(slices) != ndim:
        raise ValueError(
            f"Length of slices list ({len(slices)}) does not match number of dimensions ({ndim})."
        )

    # Check for zero-length dimensions or chunk sizes
    for dim in range(ndim):
        if shape[dim] == 0:
            raise ValueError(f"Dimension {dim} has size 0.")
        if chunks[dim] == 0:
            raise ValueError(f"Dimension {dim} has chunk size 0.")

    # Validate no_chunk_dims and convert to a set for faster lookup
    no_chunk_dims_set = set(no_chunk_dims)
    for dim in no_chunk_dims_set:
        if dim < 0 or dim >= ndim:
            raise ValueError(
                f"Dimension {dim} is out of bounds for array with {ndim} dimensions."
            )
        # Note: We no longer require that chunk size equals array size for no_chunk_dims
        # as per the updated requirement.

    # Helper function to intersect two slices
    def intersect_slices(slice1: slice, slice2: slice, dim_size: int) -> slice:
        """
        Intersect two slice objects within the bounds of a dimension.

        Parameters:
        ----------
        slice1 : slice
            The first slice.
        slice2 : slice
            The second slice.
        dim_size : int
            The size of the dimension.

        Returns:
        -------
        slice
            The intersected slice.

        Raises:
        ------
        ValueError:
            If there is no overlap between the slices.
        """

        # Convert slice to range
        def slice_to_range(s: slice) -> Tuple[int, int]:
            start, stop, step = s.indices(dim_size)
            return start, stop

        start1, stop1 = slice_to_range(slice1)
        start2, stop2 = slice_to_range(slice2)

        # Calculate intersection
        new_start = max(start1, start2)
        new_stop = min(stop1, stop2)

        if new_start >= new_stop:
            raise ValueError("No overlap between slices.")

        return slice(new_start, new_stop)

    # For each dimension, create list of slice objects
    dim_slices = []
    for dim in range(ndim):
        provided_slice = slices[dim]

        # Ensure provided_slice is a slice object
        if not isinstance(provided_slice, slice):
            raise TypeError(f"Slice for dimension {dim} is not a slice object.")

        if dim in no_chunk_dims_set:
            # Entire dimension is included; apply provided slice
            try:
                final_slice = intersect_slices(slice(None), provided_slice, shape[dim])
                dim_slices.append([final_slice])
            except ValueError:
                # No overlap; skip this chunk
                dim_slices.append([])
        else:
            # Chunked dimension: split into chunk slices and apply provided slice
            chunk_size = chunks[dim]
            dim_shape = shape[dim]
            chunk_slices = []
            for start in range(0, dim_shape, chunk_size):
                end = min(start + chunk_size, dim_shape)
                chunk_slice = slice(start, end)
                try:
                    # Intersect with provided slice
                    final_slice = intersect_slices(
                        chunk_slice, provided_slice, dim_shape
                    )
                    chunk_slices.append(final_slice)
                except ValueError:
                    # No overlap; skip this chunk
                    continue
            dim_slices.append(chunk_slices)

    # If any dimension has no slices after filtering, return empty list
    if any(len(slices_dim) == 0 for slices_dim in dim_slices):
        return []

    # Generate all combinations of slices
    all_slices = list(itertools.product(*dim_slices))
    return all_slices


def calculate_largest_slice_points(
    slice_combinations: List[Tuple[slice, ...]], shape: Tuple[int, ...]
) -> int:
    """
    Given a list of slice combinations and the shape of the Zarr array,
    this function calculates the number of points in the largest tuple of slices.

    Parameters:
    ----------
    slice_combinations : List[Tuple[slice, ...]]
        A list of slice tuples, where each tuple corresponds to a combination of slices for each dimension.

    shape : Tuple[int, ...]
        The shape of the Zarr array.

    Returns:
    -------
    int
        The number of points (elements) in the largest tuple of slices.
    """

    def calculate_points_in_tuple(slice_tuple: Tuple[slice, ...]) -> int:
        """
        Calculate the total number of elements covered by a given tuple of slices.

        Parameters:
        ----------
        slice_tuple : Tuple[slice, ...]
            A tuple of slices, one for each dimension.

        Returns:
        -------
        int
            The total number of elements covered by the tuple of slices.
        """
        total_points = 1
        for dim, s in enumerate(slice_tuple):
            # Calculate the size of the slice for this dimension
            start, stop, step = s.indices(shape[dim])
            dim_points = (stop - start) // step
            total_points *= dim_points
        return total_points

    # Calculate the points for each combination of slices
    largest_points = 0
    for slice_tuple in slice_combinations:
        points = calculate_points_in_tuple(slice_tuple)
        largest_points = max(largest_points, points)

    return largest_points
