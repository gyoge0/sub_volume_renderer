from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pygfx as gfx
import pytest
from wgpu.gui.offscreen import WgpuCanvas as OffscreenWgpuCanvas


@dataclass
class GfxContext:
    _canvas: OffscreenWgpuCanvas
    _renderer: gfx.renderers.WgpuRenderer
    _camera: gfx.OrthographicCamera
    _scene: gfx.Scene

    def _add_background(self):
        dark_gray = np.array((169, 167, 168, 255)) / 255
        light_gray = np.array((100, 100, 100, 255)) / 255
        background = gfx.Background(None, gfx.BackgroundMaterial(light_gray, dark_gray))
        self._scene.add(background)
        self._scene.add(gfx.AmbientLight())

    def render_object(self, obj: gfx.WorldObject) -> npt.NDArray[np.float32]:
        self._add_background()
        self._scene.add(obj)
        self._canvas.request_draw(
            lambda: self._renderer.render(self._scene, self._camera)
        )
        raw_result = self._canvas.draw()
        return np.asarray(raw_result)


@pytest.fixture
def gfx_context() -> GfxContext:
    canvas = OffscreenWgpuCanvas(size=(480, 480), pixel_ratio=1)
    renderer = gfx.renderers.WgpuRenderer(canvas)
    camera = gfx.OrthographicCamera()
    scene = gfx.Scene()
    return GfxContext(
        _canvas=canvas,
        _renderer=renderer,
        _camera=camera,
        _scene=scene,
    )


@pytest.fixture
def camera(gfx_context) -> gfx.OrthographicCamera:
    # noinspection PyProtectedMember
    return gfx_context._camera
