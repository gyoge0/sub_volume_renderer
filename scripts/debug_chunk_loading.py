#!/usr/bin/env python3
"""
Debug script to identify which axes and directions have chunk loading issues.
"""

import numpy as np
import pygfx as gfx
import wgpu
from funlib.geometry import Roi

from sub_volume import SubVolume, SubVolumeMaterial

# select gpu
adapters = wgpu.gpu.enumerate_adapters_sync()
description = "Quadro P2000 (DiscreteGPU) via Vulkan"
selected_adapters = [a for a in adapters if description.lower() in a.summary.lower()]
if selected_adapters:
    gfx.renderers.wgpu.select_adapter(selected_adapters[0])


def test_chunk_loading_boundaries():
    """Test chunk loading at volume boundaries to identify problematic axes/directions."""

    # Create test data - larger than typical to test boundaries
    test_data = np.random.default_rng().random(100, 80, 120).astype(np.float32)
    print(f"Test data shape: {test_data.shape}")

    # Create volume with small buffer to force frequent loading
    volume = SubVolume(
        SubVolumeMaterial(),
        data=test_data,
        buffer_shape_in_chunks=(3, 3, 3),
        chunk_shape_in_pixels=(8, 8, 8),
    )

    # Test positions at different boundaries
    test_positions = []

    # Test near each boundary in all 6 directions
    # X-axis boundaries (remember coordinate system reversal)
    test_positions.append(("X-neg boundary", (-5, 40, 60)))
    test_positions.append(("X-pos boundary", (105, 40, 60)))

    # Y-axis boundaries
    test_positions.append(("Y-neg boundary", (50, -5, 60)))
    test_positions.append(("Y-pos boundary", (50, 85, 60)))

    # Z-axis boundaries
    test_positions.append(("Z-neg boundary", (50, 40, -5)))
    test_positions.append(("Z-pos boundary", (50, 40, 125)))

    # Test center (should always work)
    test_positions.append(("Center", (50, 40, 60)))

    print(f"Volume dimensions: {volume.volume_dimensions}")
    print(f"Buffer shape in pixels: {volume.wrapping_buffer.shape_in_pixels}")
    print(f"Chunk shape in pixels: {volume.wrapping_buffer.chunk_shape_in_pixels}")

    results = []

    for desc, pos in test_positions:
        print(f"\n--- Testing {desc} at position {pos} ---")

        # Check if position can be loaded
        try:
            # Store previous ROI
            prev_roi = volume.wrapping_buffer._current_logical_roi_in_pixels

            # Try to center on position
            volume.center_on_position(pos)

            # Check if ROI was updated
            current_roi = volume.wrapping_buffer._current_logical_roi_in_pixels

            if current_roi is None:
                result = "FAILED - No ROI loaded"
            elif (
                prev_roi is not None
                and current_roi.offset == prev_roi.offset
                and current_roi.shape == prev_roi.shape
            ):
                result = "FAILED - ROI unchanged"
            else:
                result = f"SUCCESS - ROI: offset={current_roi.offset}, shape={current_roi.shape}"

        except Exception as e:
            result = f"ERROR - {str(e)}"

        results.append((desc, pos, result))
        print(f"Result: {result}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF CHUNK LOADING TESTS")
    print("=" * 80)

    failed_tests = []
    for desc, pos, result in results:
        status = "PASS" if result.startswith("SUCCESS") else "FAIL"
        print(f"{status:4} | {desc:15} | {str(pos):15} | {result}")
        if status == "FAIL":
            failed_tests.append((desc, pos, result))

    if failed_tests:
        print(f"\n{len(failed_tests)} tests failed:")
        for desc, pos, result in failed_tests:
            print(f"  - {desc} at {pos}: {result}")

        # Analyze patterns
        print("\nAnalyzing failure patterns:")
        for axis in ["X", "Y", "Z"]:
            neg_failed = any(axis + "-neg" in desc for desc, _, _ in failed_tests)
            pos_failed = any(axis + "-pos" in desc for desc, _, _ in failed_tests)
            print(f"  {axis}-axis: neg={neg_failed}, pos={pos_failed}")
    else:
        print("\nAll tests passed!")

    return failed_tests


def debug_roi_calculation(test_pos):
    """Debug the ROI calculation logic step by step."""

    test_data = np.random.default_rng().random(100, 80, 120).astype(np.float32)
    volume = SubVolume(
        SubVolumeMaterial(),
        data=test_data,
        buffer_shape_in_chunks=(3, 3, 3),
        chunk_shape_in_pixels=(8, 8, 8),
    )

    print(f"\nDebugging position {test_pos}")
    print(f"Data shape: {test_data.shape}")
    print(f"Volume dimensions: {volume.volume_dimensions}")

    # Manually calculate what center_on_position does
    size = volume.wrapping_buffer.shape_in_pixels
    print(f"Buffer size: {size}")

    # World to local space transformation
    camera_data_pos = tuple(volume.world.inverse_matrix @ np.array([*test_pos, 1]))[:3]
    print(f"Camera data pos (before reversal): {camera_data_pos}")

    # Coordinate reversal
    camera_data_pos = camera_data_pos[::-1]
    print(f"Camera data pos (after reversal): {camera_data_pos}")

    # ROI calculation
    logical_roi = Roi(
        tuple(int(c - s // 2) for c, s in zip(camera_data_pos, size)),
        size,
    )
    print(
        f"Initial logical ROI: offset={logical_roi.offset}, shape={logical_roi.shape}"
    )

    # Check bounds
    can_load = volume.wrapping_buffer.can_load_logical_roi(logical_roi)
    print(f"Can load ROI: {can_load}")

    # Get snapped ROI
    snapped_roi = volume.wrapping_buffer.get_snapped_roi_in_pixels(logical_roi)
    print(f"Snapped ROI: offset={snapped_roi.offset}, shape={snapped_roi.shape}")
    print(f"Snapped ROI empty: {snapped_roi.empty}")

    # Check data bounds intersection
    data_shape = volume.wrapping_buffer.backing_data.shape
    data_roi = Roi((0, 0, 0), data_shape)
    print(f"Data ROI: offset={data_roi.offset}, shape={data_roi.shape}")

    # Check intersection step by step
    print(
        f"Logical ROI bounds: [{logical_roi.offset}, {logical_roi.offset + logical_roi.shape})"
    )
    print(f"Data ROI bounds: [{data_roi.offset}, {data_roi.offset + data_roi.shape})")

    intersected_roi = logical_roi.intersect(data_roi)
    print(
        f"Intersected ROI: offset={intersected_roi.offset}, shape={intersected_roi.shape}"
    )
    print(f"Intersected ROI empty: {intersected_roi.empty}")


def systematic_boundary_test():
    """Test positions along each axis systematically."""

    test_data = np.random.default_rng().random(64, 64, 64).astype(np.float32)
    volume = SubVolume(
        SubVolumeMaterial(),
        data=test_data,
        buffer_shape_in_chunks=(2, 2, 2),
        chunk_shape_in_pixels=(8, 8, 8),
    )

    print(f"\nSystematic boundary test with data shape: {test_data.shape}")

    # Test along X-axis (Y=32, Z=32)
    print("\n--- X-axis test (Y=32, Z=32) ---")
    for x in range(-10, 75, 5):
        pos = (x, 32, 32)
        volume.center_on_position(pos)
        roi = volume.wrapping_buffer._current_logical_roi_in_pixels
        if roi is None:
            print(f"X={x:3d}: FAILED - No ROI")
        else:
            print(f"X={x:3d}: ROI offset={roi.offset}, shape={roi.shape}")

    # Test along Y-axis (X=32, Z=32)
    print("\n--- Y-axis test (X=32, Z=32) ---")
    for y in range(-10, 75, 5):
        pos = (32, y, 32)
        volume.center_on_position(pos)
        roi = volume.wrapping_buffer._current_logical_roi_in_pixels
        if roi is None:
            print(f"Y={y:3d}: FAILED - No ROI")
        else:
            print(f"Y={y:3d}: ROI offset={roi.offset}, shape={roi.shape}")

    # Test along Z-axis (X=32, Y=32)
    print("\n--- Z-axis test (X=32, Y=32) ---")
    for z in range(-10, 75, 5):
        pos = (32, 32, z)
        volume.center_on_position(pos)
        roi = volume.wrapping_buffer._current_logical_roi_in_pixels
        if roi is None:
            print(f"Z={z:3d}: FAILED - No ROI")
        else:
            print(f"Z={z:3d}: ROI offset={roi.offset}, shape={roi.shape}")


if __name__ == "__main__":
    print("=== Chunk Loading Boundary Tests ===")
    failed_tests = test_chunk_loading_boundaries()

    # Debug specific failing cases
    if failed_tests:
        print("\n=== ROI Calculation Debug for Failed Cases ===")
        for desc, pos, result in failed_tests[:3]:  # Debug first 3 failures
            print(f"\n--- Debugging {desc} ---")
            debug_roi_calculation(pos)

    print("\n=== Systematic Boundary Test ===")
    systematic_boundary_test()
