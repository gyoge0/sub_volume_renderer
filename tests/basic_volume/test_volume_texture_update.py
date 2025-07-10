import numpy as np

from sub_volume import Coordinate, SubVolume


def test_volume_texture_update(
    data_shape, increasing_data, chunk_dimensions, gfx_context, camera
):
    volume = SubVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=Coordinate(1, 1, 1)
    )
    volume.world.position = 0, 0, 0
    camera.show_object(volume, match_aspect=True)

    volume.ring_buffer_texture.data[:, :, :] = 0.0
    volume.ring_buffer_texture.update_full()

    expected = np.zeros((480, 480, 4))
    # Set alpha channel to opaque
    expected[:, :, 3] = 1
    # Scale to [0, 255] for comparison
    expected *= 255

    result = gfx_context.render_object(volume)

    assert np.all(expected == result)

    # update data to be all white now
    volume.ring_buffer_texture.data[:, :, :] = 1.0
    volume.ring_buffer_texture.update_full()

    expected = np.ones((480, 480, 4))
    # Scale to [0, 255] for comparison
    expected *= 255

    # we already called render_object, so the scene is set up
    result = gfx_context.draw()

    assert np.all(expected == result)
