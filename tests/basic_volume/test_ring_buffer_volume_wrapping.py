import numpy as np
from rbvr import Coordinate, GlobalSparseVolume


def test_ring_buffer_volume_wrapping(
    data_shape, increasing_data, chunk_dimensions, gfx_context, camera
):
    volume = GlobalSparseVolume(
        increasing_data, chunk_dimensions, ring_buffer_n=Coordinate(1, 1, 1)
    )
    volume.world.position = 0, 0, 0
    camera.show_object(volume, match_aspect=True)

    # Set a region to 1.0 for testing.
    # This region should be seen when wrapping!
    # Note that we probably shouldn't set to the ring buffer texture directly,
    # but this will test the wrapping in the shader.
    volume.ring_buffer_texture.data[:, :, :] = 0.0
    volume.ring_buffer_texture.data[:, 5:10, 5:10] = 1.0
    volume.ring_buffer_texture.update_full()

    expected = np.zeros((480, 480, 4))
    k = 480 // 5
    expected[k : 2 * k, k : 2 * k, :] = 1
    expected[4 * k : 5 * k, k : 2 * k, :] = 1
    expected[k : 2 * k, 4 * k : 5 * k, :] = 1
    expected[4 * k : 5 * k, 4 * k : 5 * k, :] = 1
    # Set alpha channel to opaque
    expected[:, :, 3] = 1
    # Scale to [0, 255] for comparison
    expected *= 255

    # 0, 0 of the result is at the bottom left of the screen,
    # which is index 479 in the expected array. we flip the expected
    # array to match the rendering output we should receive.
    expected = expected[::-1, :, :]

    result = gfx_context.render_object(volume)
    # result has some antialiasing along the edges, so we threshold it
    # to get a binary result for comparison.
    result = result > 127
    expected = expected > 127

    assert np.all(expected == result)
