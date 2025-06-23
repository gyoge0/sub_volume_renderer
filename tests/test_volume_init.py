import numpy as np
import pytest
from rbvr import GlobalSparseVolume


@pytest.fixture
def chunk_dimensions():
    return 5, 5, 5


@pytest.fixture
def increasing_data(chunk_dimensions):
    return np.arange(25**3).reshape((25, 25, 25)).astype(np.float32)


def test_volume_math(increasing_data, chunk_dimensions):
    volume = GlobalSparseVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=(1, 1, 1)
    )

    assert volume.volume_dimensions_in_chunks == (5, 5, 5)
    # 1 * 2 + 1
    assert volume.ring_buffer_dimensions_in_chunks == (3, 3, 3)
    # 3x3x3 chunks of 5x5x5 each
    assert volume.ring_buffer_texture.data.shape == (15, 15, 15)


def test_basic_scene_init(increasing_data, chunk_dimensions, gfx_context, caplog):
    volume = GlobalSparseVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=(2, 2, 2)
    )
    with caplog.at_level("ERROR", logger="wgpu"):
        _ = gfx_context.render_object(volume)
    assert caplog.text == ""


def test_volume_positioning(increasing_data, chunk_dimensions, gfx_context, camera):
    volume = GlobalSparseVolume(increasing_data, chunk_dimensions)
    volume.world.position = 0, 0, 0
    camera.show_object(volume, match_aspect=True)

    result = gfx_context.render_object(volume)
    assert np.all(result == 255.0)
