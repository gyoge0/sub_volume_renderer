# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated volume renderer built on top of pygfx/WebGPU. The renderer is designed for efficient visualization of large volumetric datasets using a wrapping buffer system and raycasting with Maximum Intensity Projection (MIP).

## Development Commands

### Build and Run
- `pixi run main` - Run the main volume renderer demo
- `pixi run test` - Run the test suite
- `pixi run check` - Run ruff linting checks
- `pixi run format` - Auto-format code with ruff (includes F401 unused imports and isort)

### Development Environment
- Uses pixi for dependency management
- Python 3.11+ required
- GPU support required (WebGPU/Vulkan)

## Core Architecture

### Volume Rendering Pipeline
1. **SubVolume** (`_wobject.py`): Main volume object that wraps data and manages rendering
2. **WrappingBuffer** (`_wrapping_buffer.py`): Handles chunked data loading and memory management
3. **SubVolumeShader** (`_shader.py`): WebGPU shader integration with pygfx
4. **WGSL Shaders** (`shaders/`): GPU compute shaders for raycasting

### Key Components

#### WrappingBuffer System
- Manages large volumetric datasets through chunked loading
- Uses a ring buffer in GPU memory for efficient data streaming
- Handles coordinate system transformations (Fortran vs C-style ordering)
- **Critical**: Shader coordinates are Fortran-style (z,y,x) while Python uses C-style (x,y,z)
- Array reversals `[::-1]` are intentional for coordinate system consistency

#### Coordinate Systems
- **Python/NumPy**: C-style ordering (x,y,z)
- **Shader/GPU**: Fortran-style ordering (z,y,x)
- All coordinate transformations between Python and shaders require `[::-1]` reversals
- The `volume_dimensions` and buffer offsets are automatically reversed when passed to shaders

#### Shader Pipeline
- **fs_main.wgsl**: Fragment shader handling ray-volume intersection and depth calculation
- **sample_vol.wgsl**: Volume sampling with bounds checking and wrapping buffer access
- **raycast.wgsl**: Maximum Intensity Projection raycasting implementation
- **volume_common.wgsl**: Common volume rendering utilities

### Data Flow
1. Large volumetric data (numpy/zarr arrays) → WrappingBuffer
2. WrappingBuffer loads chunks into GPU texture memory
3. Camera position changes trigger `center_on_position()` to update loaded regions
4. Shader raycasts through volume finding maximum intensity voxels
5. Depth calculation and fragment output for proper compositing

## Known Issues and Debugging

### Camera Inside Volume
When the camera is positioned inside the volume, depth calculation can become unstable due to:
- Very small `ndc_pos.w` values (view-space distance)
- Division by near-zero causing fragment discard during depth testing
- **Fix**: Use `clamp(ndc_pos.z / max(ndc_pos.w, 0.0001), 0.0, 1.0)` for robust depth calculation

### Boundary Chunk Loading (RESOLVED)
Previously, chunks wouldn't load at the "far end" of data boundaries on all axes. This was caused by:
- `get_snapped_roi_in_pixels` using intersection+shrink which created empty ROIs at boundaries
- **Root cause**: intersection+shrink mode after grid snapping removed chunks that partially extended beyond data bounds
- **Solution**: Changed to intersection+grow mode, allowing partial out-of-bounds chunks to load
- **Key insight**: numpy/zarr arrays safely handle out-of-bounds access by returning smaller shapes
- **Fix location**: `_wrapping_buffer.py` lines 80-99 and `load_into_buffer` method lines 212-252

### Coordinate System Debugging
- If shader sampling appears offset, check coordinate system reversals
- `current_logical_offset_in_pixels` and `current_logical_shape_in_pixels` are passed in Fortran order
- Volume dimensions are reversed when passed to uniform buffers

## Testing

### Test Structure
- `tests/basic_volume/` - Core volume rendering tests
- `tests/wrapping_buffer/` - Wrapping buffer unit tests
  - `test_boundary_loading.py` - Comprehensive boundary loading edge cases (21 tests)
- Uses pytest with offscreen WebGPU canvas for headless testing
- `conftest.py` provides `gfx_context` fixture for rendering tests

### Running Specific Tests
```bash
pixi run test tests/basic_volume/test_volume_init.py
pixi run test tests/wrapping_buffer/test_load_logical_roi.py
pixi run test tests/wrapping_buffer/test_boundary_loading.py
```

## Dependencies

### Core Dependencies
- `pygfx` - WebGPU-based graphics library
- `numpy` - Numerical computing
- `zarr` - Chunked array storage
- `funlib.geometry` - Geometry utilities (Roi, Coordinate)
- `wgpu` - WebGPU bindings

### Development Dependencies
- `pytest` - Testing framework
- `hypothesis` - Property-based testing
- `ruff` - Linting and formatting

## Performance Considerations

### Memory Management
- WrappingBuffer maintains a fixed-size GPU texture buffer
- Only loads necessary chunks based on camera position
- Automatic wrapping and reuse of buffer space

### GPU Optimization
- Uses textureLoad for direct texture access (no interpolation)
- Raycasting with refinement step for accurate maximum finding
- Bounds checking prevents unnecessary sampling outside data regions

## Development Tips

### Shader Development
- WGSL shaders use Jinja2 templating via pygfx
- Shader compilation errors appear in WebGPU validation
- Use `textureLoad` for exact voxel sampling, not `textureSample`

### Debugging Volume Rendering
- Check `current_logical_offset_in_pixels` and `current_logical_shape_in_pixels` in uniform buffer
- Verify camera position with `center_on_position()` calls
- Use debug colors in shaders to visualize bounds and sampling issues

### Debugging Chunk Loading Issues
- Use `scripts/debug_chunk_loading.py` to systematically test boundary loading
- Check if `get_snapped_roi_in_pixels` returns empty ROIs at boundaries
- Verify `load_into_buffer` handles partial out-of-bounds data correctly
- Test with various chunk sizes and buffer configurations