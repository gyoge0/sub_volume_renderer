import numpy as np
from funlib.geometry import Roi

# todo: use the chunk size fixture somehow and don't hardcode chunk size as 4?


def test_load_into_buffer_basic(buffer):
    buffer_roi = Roi((0, 0, 0), (1, 1, 1))
    logical_roi = Roi((0, 0, 0), (1, 1, 1))

    buffer.load_into_buffer(buffer_roi, logical_roi)

    written_data = buffer.texture.data[0:4, 0:4, 0:4]
    source_data = buffer.backing_data[0:4, 0:4, 0:4]

    np.testing.assert_array_equal(source_data, written_data)


def test_load_into_buffer_offset(buffer):
    # we are writing to 0:4, 0:4, 0:4 since our chunk size is 4
    buffer_roi = Roi((0, 0, 0), (1, 1, 1))
    # we are reading from 4:8, 4:8, 4:8 since our backing data is offset
    logical_roi = Roi((1, 1, 1), (1, 1, 1))

    buffer.load_into_buffer(buffer_roi, logical_roi)

    written_data = buffer.texture.data[0:4, 0:4, 0:4]
    source_data = buffer.backing_data[4:8, 4:8, 4:8]

    np.testing.assert_array_equal(source_data, written_data)
