import numpy as np
from funlib.geometry import Coordinate

from sub_volume import SubVolume


def test_volume_math(increasing_data, chunk_dimensions):
    volume = SubVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=Coordinate(1, 1, 1)
    )

    assert volume.volume_dimensions_in_chunks == (5, 5, 5)
    # 1 * 2 + 1
    assert volume.ring_buffer_shape_in_chunks == (3, 3, 3)
    # 3x3x3 chunks of 5x5x5 each
    assert volume.ring_buffer_texture.data.shape == (15, 15, 15)


def test_basic_scene_init(increasing_data, chunk_dimensions, gfx_context, caplog):
    volume = SubVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=Coordinate(5, 5, 5)
    )
    with caplog.at_level("ERROR", logger="wgpu"):
        _ = gfx_context.render_object(volume)
    assert caplog.text == ""


def test_volume_positioning(increasing_data, chunk_dimensions, gfx_context, camera):
    volume = SubVolume(increasing_data, chunk_dimensions)
    volume.world.position = 0, 0, 0
    camera.show_object(volume, match_aspect=True)

    result = gfx_context.render_object(volume)
    assert np.all(result == 255.0)
