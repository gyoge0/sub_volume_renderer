# entire file written by Claude Sonnet 4 with Claude Code
# watch out for issues, and be skeptical of the accuracy of the code
"""
Script to analyze what happens with non-chunk-aligned ROIs and out-of-bounds data access.
"""

import numpy as np
from funlib.geometry import Roi

from sub_volume._wrapping_buffer import WrappingBuffer


def analyze_non_chunk_aligned_roi():
    """Analyze what happens when ROI size is not a multiple of chunk size."""

    # Create small test data
    backing_data = np.arange(20 * 20 * 20).reshape(20, 20, 20).astype(np.float32)

    # Create buffer with chunk_size=8
    buffer = WrappingBuffer(
        backing_data=backing_data,
        shape_in_chunks=(2, 2, 2),
        chunk_shape_in_pixels=(8, 8, 8),
    )

    print("=== Analyzing Non-Chunk-Aligned ROI ===")
    print(f"Chunk size: {buffer.chunk_shape_in_pixels}")
    print(f"Buffer size: {buffer.shape_in_pixels}")

    # Try to load ROI with size 7 (not multiple of chunk_size=8)
    roi = Roi((0, 0, 0), (7, 7, 7))
    print(f"Original ROI: offset={roi.offset}, shape={roi.shape}")

    try:
        snapped_roi = buffer.get_snapped_roi_in_pixels(roi)
        print(f"Snapped ROI: offset={snapped_roi.offset}, shape={snapped_roi.shape}")
        print(f"Snapped ROI empty: {snapped_roi.empty}")

        if not snapped_roi.empty:
            buffer.load_logical_roi(roi)
            print("Load succeeded")
            print(f"Current ROI: {buffer._current_logical_roi_in_pixels}")
        else:
            print("Snapped ROI is empty")

    except Exception as e:
        print(f"Error: {e}")


def analyze_out_of_bounds_data_access():
    """Analyze what happens when trying to access data beyond backing array bounds."""

    # Create small test data - only 10x10x10
    backing_data = np.arange(10 * 10 * 10).reshape(10, 10, 10).astype(np.float32)

    # Create buffer that could theoretically hold more
    buffer = WrappingBuffer(
        backing_data=backing_data,
        shape_in_chunks=(3, 3, 3),
        chunk_shape_in_pixels=(4, 4, 4),
    )

    print("\n=== Analyzing Out-of-Bounds Data Access ===")
    print(f"Backing data shape: {backing_data.shape}")
    print(f"Buffer size: {buffer.shape_in_pixels}")

    # Try to load ROI that extends beyond backing data
    roi = Roi(
        (8, 8, 8), (4, 4, 4)
    )  # This would access indices 8-11, but data only goes 0-9
    print(f"Out-of-bounds ROI: offset={roi.offset}, shape={roi.shape}")

    try:
        snapped_roi = buffer.get_snapped_roi_in_pixels(roi)
        print(f"Snapped ROI: offset={snapped_roi.offset}, shape={snapped_roi.shape}")
        print(f"Snapped ROI empty: {snapped_roi.empty}")

        if not snapped_roi.empty:
            buffer.load_logical_roi(roi)
            print("Load succeeded")
            print(f"Current ROI: {buffer._current_logical_roi_in_pixels}")
        else:
            print("Snapped ROI is empty")

    except Exception as e:
        print(f"Error: {e}")


def analyze_partially_out_of_bounds():
    """Analyze what happens with ROI that's partially outside backing data."""

    # Create small test data
    backing_data = np.arange(10 * 10 * 10).reshape(10, 10, 10).astype(np.float32)

    buffer = WrappingBuffer(
        backing_data=backing_data,
        shape_in_chunks=(3, 3, 3),
        chunk_shape_in_pixels=(4, 4, 4),
    )

    print("\n=== Analyzing Partially Out-of-Bounds ROI ===")
    print(f"Backing data shape: {backing_data.shape}")

    # ROI that starts inside but extends outside
    roi = Roi(
        (6, 6, 6), (8, 8, 8)
    )  # This would access indices 6-13, but data only goes 0-9
    print(f"Partially out-of-bounds ROI: offset={roi.offset}, shape={roi.shape}")

    try:
        snapped_roi = buffer.get_snapped_roi_in_pixels(roi)
        print(f"Snapped ROI: offset={snapped_roi.offset}, shape={snapped_roi.shape}")
        print(f"Snapped ROI empty: {snapped_roi.empty}")

        if not snapped_roi.empty:
            buffer.load_logical_roi(roi)
            print("Load succeeded")
            print(f"Current ROI: {buffer._current_logical_roi_in_pixels}")
        else:
            print("Snapped ROI is empty")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    analyze_non_chunk_aligned_roi()
    analyze_out_of_bounds_data_access()
    analyze_partially_out_of_bounds()
