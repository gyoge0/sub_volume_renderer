# Subvolume Renderer

This project is a volume renderer built with Pygfx and WGPU designed for
handling large electron microscopy datasets. It internally uses multiple 
3D ring buffers to load areas around the camera and implements 
level-of-detail to render larger areas on lower-end systems.

# Usage

```py
import pygfx as gfx
from sub_volume import SubVolume, SubVolumeMaterial

# create a Pygfx scene, canvas, and renderer
scene, canvas, renderer = ...

# load in data_{0,1,2} and segmentations_{0,1,2}
# SubVolume supports numpy, zarr, and tensorstore arrays
# take a look at `scripts/multi_scale.py` for a quick example
data_0, segmentations_0 = ...
data_1, segmentations_1 = ...
data_1, segmentations_1 = ...

# create our volume
volume = SubVolume(
    SubVolumeMaterial(
        # the minimum intensity threshold for the raycasting algorithim
        # see https://en.wikipedia.org/wiki/Local_maximum_intensity_projection
        lmip_threshold=0.5,
        # higher density is more fog
        fog_density=0.01,
        # the hsv colors available for segmentation labels
        # the color for a label will be `label_id % len(colors)`
        colors=[
            (0.0, 1.0, 1.0), # red
            (0.33, 1.0, 1.0), # green
            (0.66, 1.0, 1.0), # blue
        ],
    ),
    data_segmentation_pairs=[
        (data_0, segmentations_0),
        (data_1, segmentations_1),
        (data_2, segmentations_2),
    ],
    # the shape of a chunk for each level in pixels
    # (8, 8, 48) means we might load data_1[:8, :8, :48]
    chunk_shape_in_pixels=[
        (16, 16, 48),
        (8, 8, 48),
        (4, 4, 48),
    ],
    # the shape of the buffer for each level in pixels
    buffer_shape_in_chunks=[
        (2, 2, 2),
        (4, 4, 4),
        (8, 8, 8),
    ],
)
scene.add(volume)

# center our selection on the camera every frame
@canvas.request_draw
def do_draw():
    # render a frame
    renderer.render(scene, camera)
    # center the SubVolume on our camera
    volume.center_on_position(camera.world.position)
```

# Development

Install [Pixi](https://pixi.sh/latest/). Then, clone the repo and run 
`pixi run python scripts/main.py` to get started. You might need to
install glfw with your system's package manager.

## Janelia Datasets

`platynereis.py`, `create_platynereis_multiscale`, `mouse.py`, and 
`create_mouse_multiscale` attempt to read the Funke Lab's datasets from 
the Janelia filesystem (paths in `/nrs/funke`). Contact [Jan 
Funke](mailto:funkej@hhmi.org) or [Caroline 
Malin-Mayor](mailto:malinmayorc@hhmi.org) with questions. All other 
scripts should be runnable outside Janelia (namely `main.py` and 
`multi_scale.py`).

## GPU Selection

Several of the scripts include the following codeblock to select a
specific GPU:

```py
# select gpu
adapters = wgpu.gpu.enumerate_adapters_sync()
description = "Quadro P2000 (DiscreteGPU) via Vulkan"
selected_adapters = [a for a in adapters if description.lower() in a.summary.lower()]
if selected_adapters:
    gfx.renderers.wgpu.select_adapter(selected_adapters[0])
```

This fixes the following error you might get on dual-GPU systems:

```
thread '<unnamed>' panicked at src/lib.rs:598:5:
Error in wgpuSurfaceConfigure: Validation Error

Caused by:
  Surface does not support the adapter's queue family

note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
fatal runtime error: failed to initiate panic, error 5
```

<!-- 
Funke Lab note: this error was only found on funkelab-ws2 which has a
TITAN XP and a Quadro P2000
-->
