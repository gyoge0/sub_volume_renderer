import numpy as np
import pygfx as gfx
import wgpu
import zarr
from rbvr import Coordinate, GlobalSparseVolume, Roi, WorldCoordinateRingBufferManager
from rendercanvas.auto import RenderCanvas, loop

# select gpu
adapters = wgpu.gpu.enumerate_adapters_sync()
description = "Quadro P2000 (DiscreteGPU) via Vulkan"
selected_adapters = [a for a in adapters if description.lower() in a.summary.lower()]
if selected_adapters:
    gfx.renderers.wgpu.select_adapter(selected_adapters[0])

# noinspection SpellCheckingInspection
data = zarr.open_array(
    "/nrs/funke/data/lightsheet/130312_platynereis/13-03-12.zarr/raw"
)
# data is stored as [channel, t, z, y, x]
scaled_data = data[0, 378, :, :, :]
scaled_data = scaled_data / scaled_data.max()
scaled_data[:25, :25, :25] = 1.0
scaled_data = scaled_data.astype(np.float32)

canvas = RenderCanvas(size=(480, 480))
renderer = gfx.renderers.WgpuRenderer(canvas)
camera = gfx.PerspectiveCamera(fov=45)
controller = gfx.FlyController(camera=camera, register_events=renderer)
scene = gfx.Scene()

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
# noinspection PyTypeChecker
background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
scene.add(background)
scene.add(gfx.AmbientLight())

chunk_dimensions = Coordinate(256, 64, 64)
view_distance = Coordinate(3, 3, 3)
volume = GlobalSparseVolume(
    scaled_data,
    chunk_dimensions,
    view_distance,
)

volume.ring_buffer_texture.data[:, :, :] = np.zeros(
    (view_distance * 2 + 1) * chunk_dimensions, dtype=np.float32
)
volume.ring_buffer_texture.update_full()

volume.world.position = 0, 0, 0

scene.add(volume)

manager = WorldCoordinateRingBufferManager(
    chunk_dimensions=chunk_dimensions,
    view_distance=view_distance,
    volume_dimensions=scaled_data.shape,
    initial_camera_coordinate=camera.world.position,
)


def update_ops(ops):
    for dst, src in ops:
        copy_roi = Roi(
            offset=dst.offset,
            shape=scaled_data[src.to_slices()].shape,
        )
        volume.ring_buffer_texture.data[dst.to_slices()] = 0.0
        volume.ring_buffer_texture.data[copy_roi.to_slices()] = scaled_data[
            src.to_slices()
        ]
        volume.ring_buffer_texture.update_range(copy_roi.offset, copy_roi.shape)


update_ops(manager.initial_assignments)


@canvas.request_draw
def do_draw():
    renderer.render(scene, camera)
    # our camera pos is in world coordinates
    # we convert it into volume data coordinates
    camera_data_pos = tuple(
        volume.world.inverse_matrix @ camera.world.matrix @ np.array([0, 0, 0, 1])
    )[:3]
    # I hate pygfx and wgpu and everything related to indexing never ever touch it on your own
    # the camera position matches up with the Fortran style stuff? is this why the other position uniform
    # things just magically correct in the shader?
    camera_data_pos = camera_data_pos[::-1]

    # noinspection PyTypeChecker
    ops: set[tuple[Roi, Roi]] = manager.move_camera(camera_data_pos)
    update_ops(ops)


camera.show_object(volume, view_dir=(-1, 0, 0))

loop.run()
