# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "glfw",
#     "pygfx",
# ]
# ///

import time
from threading import Thread

import numpy as np
import pygfx as gfx
import wgpu
from wgpu.gui.auto import WgpuCanvas, run

TEXTURE_WIDTH = 8 * 1024
TEXTURE_HEIGHT = 8 * 1024

canvas = WgpuCanvas(size=(512, 512), max_fps=999)
renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
scene = gfx.Scene()
camera = gfx.OrthographicCamera(TEXTURE_WIDTH, TEXTURE_HEIGHT)
camera.local.y = TEXTURE_HEIGHT / 2
camera.local.scale_y = -1
camera.local.x = TEXTURE_WIDTH / 2

colormap1 = gfx.cm.plasma


def create_random_image():
    rand_img = np.random.rand(TEXTURE_WIDTH, TEXTURE_HEIGHT).astype(np.float32) * 255

    return gfx.Image(
        gfx.Geometry(grid=gfx.Texture(rand_img, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255), map=colormap1),
    )


image = create_random_image()
image.local.x = 0
image.local.y = 0
scene.add(image)


device = renderer.device
queue = device.queue

for adapter in wgpu.gpu.enumerate_adapters_sync():
    print()
    print(adapter.summary)


last_log_s = int(time.time())


def update_texture(data):
    global last_log_s

    # to simulate a lot of writes
    repetitions = 10
    time_set = 0
    time_queue = 0
    for i in range(repetitions):
        start = time.time()
        image.geometry.grid.data[:] = data
        image.geometry.grid.update_range((0, 0, 0), image.geometry.grid.size)
        time_set += time.time() - start

        # as per https://github.com/gfx-rs/wgpu/discussions/5525#discussioncomment-9267732
        # A lot of time is spent in here, so I guess this is where the actual
        # data transfer happens. At least on a Metal backend, this seems to
        # block the render thread.
        start = time.time()
        queue.submit([])
        time_queue += time.time() - start

    now_s = int(time.time())
    if now_s > last_log_s:
        print(f"\tset data in {time_set:.3f}s")
        print(f"\tsubmitted queue in {time_queue:.3f}s")
        last_log_s = now_s


running = True

print("Creating data...")
data = np.random.rand(10, TEXTURE_WIDTH, TEXTURE_HEIGHT).astype(np.float32) * 255
print("...done!")


def update_texture_loop():
    while running:
        update_texture(data[int(time.time() * 1000) % 10])


def redraw():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    reload_thread = Thread(target=update_texture_loop)
    #######################################################
    # comment out the next line to test rendering without background loading
    # (that'll give you the raw rendering performance)
    # then compare with background loading and see if it gets slower
    #######################################################
    # reload_thread.start()
    canvas.request_draw(redraw)
    run()
    running = False
    reload_thread.join()


#######################
#                     #
#       RESULTS       #
#                     #
#######################

# Commenting out the line to turn off background threads gets 60 fps and
# max performance. Uncommenting it lowers it to something like 3 fps.
# This suggests that running a background thread just makes it slower
# that it's best to run single threaded?
