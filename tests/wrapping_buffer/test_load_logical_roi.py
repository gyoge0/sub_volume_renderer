import numpy as np
from funlib.geometry import Roi


def test_load_logical_roi_full(buffer):
    roi = Roi((0, 0, 0), (20, 20, 20))
    buffer.load_logical_roi(roi)
    # The entire texture should match the backing array
    np.testing.assert_array_equal(
        buffer.texture.data, buffer.backing_data[:20, :20, :20]
    )


def test_load_logical_roi_chunk(buffer):
    roi = Roi((0, 0, 0), (4, 4, 4))

    buffer.load_logical_roi(roi)
    # Only the first chunk should be filled
    expected = buffer.backing_data[0:4, 0:4, 0:4]
    actual = buffer.texture.data[0:4, 0:4, 0:4]
    np.testing.assert_array_equal(actual, expected)
    # The rest should be zero
    assert np.all(buffer.texture.data[4:8, :, :] == 0)
    assert np.all(buffer.texture.data[:, 4:8, :] == 0)
    assert np.all(buffer.texture.data[:, :, 4:8] == 0)


def test_load_logical_roi_offset_chunk(buffer):
    roi = Roi((4, 4, 4), (4, 4, 4))

    buffer.load_logical_roi(roi)
    # Only the offset chunk should be filled
    expected = buffer.backing_data[4:8, 4:8, 4:8]
    actual = buffer.texture.data[4:8, 4:8, 4:8]
    np.testing.assert_array_equal(actual, expected)
    # The rest should be zero
    assert np.all(buffer.texture.data[0:4, :, :] == 0)
    assert np.all(buffer.texture.data[:, 0:4, :] == 0)
    assert np.all(buffer.texture.data[:, :, 0:4] == 0)


def test_load_logical_roi_empty(buffer):
    empty_roi = Roi((0, 0, 0), (0, 0, 0))
    buffer.load_logical_roi(empty_roi)
    assert np.all(buffer.texture.data == 0)


def test_load_logical_roi_disjoint(buffer):
    roi1 = Roi((0, 0, 0), (4, 4, 4))
    roi2 = Roi((4, 4, 4), (4, 4, 4))
    buffer.load_logical_roi(roi1)
    buffer.load_logical_roi(roi2)
    expected1 = buffer.backing_data[0:4, 0:4, 0:4]
    expected2 = buffer.backing_data[4:8, 4:8, 4:8]
    np.testing.assert_array_equal(buffer.texture.data[0:4, 0:4, 0:4], expected1)
    np.testing.assert_array_equal(buffer.texture.data[4:8, 4:8, 4:8], expected2)


def test_load_logical_roi_partial_overlap(buffer):
    roi1 = Roi((0, 0, 0), (4, 4, 4))
    roi2 = Roi((2, 0, 0), (4, 4, 4))
    buffer.load_logical_roi(roi1)
    buffer.load_logical_roi(roi2)
    expected = buffer.backing_data[2:6, 0:4, 0:4]
    actual = buffer.texture.data[2:6, 0:4, 0:4]
    np.testing.assert_array_equal(actual, expected)


def test_load_logical_roi_already_loaded(buffer):
    roi = Roi((0, 0, 0), (4, 4, 4))
    buffer.load_logical_roi(roi)
    # Load again, should not change
    before = buffer.texture.data.copy()
    buffer.load_logical_roi(roi)
    after = buffer.texture.data
    np.testing.assert_array_equal(before, after)
