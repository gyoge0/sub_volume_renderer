from funlib.geometry import Roi


def test_can_load_logical_roi_inside(buffer):
    roi = Roi((0, 0, 0), (4, 4, 4))
    assert buffer.can_load_logical_roi(roi)


def test_can_load_logical_roi_exact_fit(buffer):
    roi = Roi((0, 0, 0), (20, 20, 20))
    assert buffer.can_load_logical_roi(roi)


def test_can_load_logical_roi_too_large(buffer):
    roi = Roi((0, 0, 0), (100, 100, 100))
    assert not buffer.can_load_logical_roi(roi)


def test_can_load_logical_roi_partial_outside(buffer):
    # even though the Roi is outside the data, this method shouldn't check for that
    roi = Roi((39, 39, 39), (4, 4, 4))
    assert buffer.can_load_logical_roi(roi)


def test_can_load_logical_roi_empty(buffer):
    roi = Roi((0, 0, 0), (0, 0, 0))
    assert buffer.can_load_logical_roi(roi)


def test_can_load_logical_roi_offset_beyond(buffer):
    # even though the Roi is outside the data, this method shouldn't check for that
    roi = Roi((41, 41, 41), (1, 1, 1))
    assert buffer.can_load_logical_roi(roi)
