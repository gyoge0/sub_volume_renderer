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
camera = gfx.OrthographicCamera()
scene = gfx.Scene()

dark_gray = np.array((169, 167, 168, 255)) / 255
light_gray = np.array((100, 100, 100, 255)) / 255
# noinspection PyTypeChecker
background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
scene.add(background)
scene.add(gfx.AmbientLight())

increasing_data = np.arange(25**3).reshape((25, 25, 25)).astype(np.float32)
chunk_dimensions = (5, 5, 5)
volume = GlobalSparseVolume(increasing_data, chunk_dimensions)
volume.world.position = 0, 0, 0

camera.show_object(volume, match_aspect=True)

scene.add(volume)
canvas.request_draw(lambda: renderer.render(scene, camera))
loop.run()
