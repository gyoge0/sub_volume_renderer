# entire file written by Claude Sonnet 4 with Claude Code
# watch out for issues, and be skeptical of the accuracy of the code
"""
Script to analyze what happens when accessing out-of-bounds data in numpy and zarr arrays.
"""

import numpy as np
import zarr


def analyze_numpy_out_of_bounds():
    """Analyze what happens with numpy array out-of-bounds access."""

    print("=== Analyzing NumPy Out-of-Bounds Access ===")

    # Create small numpy array
    arr = np.arange(10 * 10 * 10).reshape(10, 10, 10).astype(np.float32)
    print(f"Array shape: {arr.shape}")

    # Test different out-of-bounds scenarios
    scenarios = [
        ("Completely out of bounds", slice(15, 20)),
        ("Partially out of bounds", slice(8, 15)),
        ("Negative indices", slice(-5, 3)),
        ("Start beyond end", slice(12, 8)),
    ]

    for desc, test_slice in scenarios:
        print(f"\n{desc}: arr[{test_slice}, 0, 0]")
        try:
            result = arr[test_slice, 0, 0]
            print(f"  Result shape: {result.shape}")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def analyze_zarr_out_of_bounds():
    """Analyze what happens with zarr array out-of-bounds access."""

    print("\n=== Analyzing Zarr Out-of-Bounds Access ===")

    # Create small zarr array
    arr = zarr.array(np.arange(10 * 10 * 10).reshape(10, 10, 10).astype(np.float32))
    print(f"Array shape: {arr.shape}")

    # Test different out-of-bounds scenarios
    scenarios = [
        ("Completely out of bounds", slice(15, 20)),
        ("Partially out of bounds", slice(8, 15)),
        ("Negative indices", slice(-5, 3)),
        ("Start beyond end", slice(12, 8)),
    ]

    for desc, test_slice in scenarios:
        print(f"\n{desc}: arr[{test_slice}, 0, 0]")
        try:
            result = arr[test_slice, 0, 0]
            print(f"  Result shape: {result.shape}")
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def analyze_multi_dimensional_out_of_bounds():
    """Analyze multi-dimensional out-of-bounds access."""

    print("\n=== Analyzing Multi-Dimensional Out-of-Bounds Access ===")

    # Test with numpy
    arr = np.arange(5 * 5 * 5).reshape(5, 5, 5).astype(np.float32)
    print(f"NumPy array shape: {arr.shape}")

    test_slices = [
        (slice(3, 8), slice(0, 3), slice(0, 3)),  # First dim out of bounds
        (slice(0, 3), slice(3, 8), slice(0, 3)),  # Second dim out of bounds
        (slice(0, 3), slice(0, 3), slice(3, 8)),  # Third dim out of bounds
        (slice(6, 10), slice(6, 10), slice(6, 10)),  # All dims out of bounds
    ]

    for i, slices in enumerate(test_slices, 1):
        print(f"\nScenario {i}: arr[{slices[0]}, {slices[1]}, {slices[2]}]")
        try:
            result = arr[slices]
            print(f"  Result shape: {result.shape}")
            print(f"  Result dtype: {result.dtype}")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    analyze_numpy_out_of_bounds()
    analyze_zarr_out_of_bounds()
    analyze_multi_dimensional_out_of_bounds()
