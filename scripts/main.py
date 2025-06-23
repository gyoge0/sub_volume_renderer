import numpy as np
import pygfx as gfx
import wgpu
from rbvr import GlobalSparseVolume
from rendercanvas.auto import RenderCanvas, loop

# select gpu
adapters = wgpu.gpu.enumerate_adapters_sync()
description = "Quadro P2000 (DiscreteGPU) via Vulkan"
selected_adapters = [a for a in adapters if description.lower() in a.summary.lower()]
if selected_adapters:
    gfx.renderers.wgpu.select_adapter(selected_adapters[0])

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

chunk_dimensions = (5, 5, 5)
volume = GlobalSparseVolume(
    np.zeros((25, 25, 25), dtype=np.float32),
    chunk_dimensions,
    (1, 1, 1),
)

initial_data = np.zeros((15, 15, 15))
initial_data[:, :, :] = 0.0
initial_data[:, 5:10, 5:10] = 1.0
initial_data = initial_data.astype(np.float32)

volume.ring_buffer_texture.data[:, :, :] = initial_data
volume.ring_buffer_texture.update_full()

volume.world.position = 0, 0, 0

camera.show_object(volume, match_aspect=True)

scene.add(volume)
canvas.request_draw(lambda: renderer.render(scene, camera))
loop.run()
