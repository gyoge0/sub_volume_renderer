from funlib.geometry import Roi


def test_no_wrap(buffer):
    roi = Roi((0, 0, 0), (2, 2, 2))
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 1
    buffer_roi, logical_roi = result[0]
    assert buffer_roi.offset == (0, 0, 0)
    assert buffer_roi.shape == (2, 2, 2)
    assert logical_roi == roi


def test_wrap_one_dim(buffer):
    roi = Roi((4, 0, 0), (3, 3, 3))  # crosses boundary in dim 0
    # actual:
    # 0 1 2 3 4 5 6 8 9
    #         X|X X
    # expected:
    # 0 1 2 3 4 5 6 8 9
    # X X     X
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 2
    assert (Roi((4, 0, 0), (1, 3, 3)), Roi((4, 0, 0), (1, 3, 3))) in result
    assert (Roi((0, 0, 0), (2, 3, 3)), Roi((5, 0, 0), (2, 3, 3))) in result


def test_wrap_two_dims(buffer):
    roi = Roi((3, 2, 0), (4, 4, 4))
    # actual first:
    # 0 1 2 3 4 5 6 7 8 9
    #       X X|X X
    # expected:
    # 0 1 2 3 4 5 6 7 8 9
    # X X     X X

    # actual second:
    # 0 1 2 3 4 5 6 7 8 9
    #     X X X|X
    # expected:
    # 0 1 2 3 4 5 6 7 8 9
    # X   X X X
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 4
    assert (Roi((3, 2, 0), (2, 3, 4)), Roi((3, 2, 0), (2, 3, 4))) in result
    assert (Roi((0, 2, 0), (2, 3, 4)), Roi((5, 2, 0), (2, 3, 4))) in result
    assert (Roi((3, 0, 0), (2, 1, 4)), Roi((3, 5, 0), (2, 1, 4))) in result
    assert (Roi((0, 0, 0), (2, 1, 4)), Roi((5, 5, 0), (2, 1, 4))) in result


def test_wrap_three_dims(buffer):
    roi = Roi((3, 2, 4), (4, 4, 4))
    # actual first:
    # 0 1 2 3 4 5 6 7 8 9
    #       X X|X X
    # expected:
    # 0 1 2 3 4 5 6 7 8 9
    # X X     X X

    # actual second:
    # 0 1 2 3 4 5 6 7 8 9
    #     X X X|X
    # expected:
    # 0 1 2 3 4 5 6 7 8 9
    # X   X X X

    # actual third:
    # 0 1 2 3 4 5 6 7 8 9
    #         X|X X X
    # expected:
    # 0 1 2 3 4 5 6 7 8 9
    # X X X   X
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 8
    assert (Roi((3, 2, 4), (2, 3, 1)), Roi((3, 2, 4), (2, 3, 1))) in result
    assert (Roi((0, 2, 4), (2, 3, 1)), Roi((5, 2, 4), (2, 3, 1))) in result
    assert (Roi((3, 0, 4), (2, 1, 1)), Roi((3, 5, 4), (2, 1, 1))) in result
    assert (Roi((0, 0, 4), (2, 1, 1)), Roi((5, 5, 4), (2, 1, 1))) in result
    assert (Roi((3, 2, 0), (2, 3, 3)), Roi((3, 2, 5), (2, 3, 3))) in result
    assert (Roi((0, 2, 0), (2, 3, 3)), Roi((5, 2, 5), (2, 3, 3))) in result
    assert (Roi((3, 0, 0), (2, 1, 3)), Roi((3, 5, 5), (2, 1, 3))) in result
    assert (Roi((0, 0, 0), (2, 1, 3)), Roi((5, 5, 5), (2, 1, 3))) in result


def test_wrap_three_dims_multiple_wraps(buffer):
    # same as test_wrap_three_dims except we are not in the origin buffer buffer
    roi = Roi((13, 12, 14), (4, 4, 4))
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 8
    assert (Roi((3, 2, 4), (2, 3, 1)), Roi((13, 12, 14), (2, 3, 1))) in result
    assert (Roi((0, 2, 4), (2, 3, 1)), Roi((15, 12, 14), (2, 3, 1))) in result
    assert (Roi((3, 0, 4), (2, 1, 1)), Roi((13, 15, 14), (2, 1, 1))) in result
    assert (Roi((0, 0, 4), (2, 1, 1)), Roi((15, 15, 14), (2, 1, 1))) in result
    assert (Roi((3, 2, 0), (2, 3, 3)), Roi((13, 12, 15), (2, 3, 3))) in result
    assert (Roi((0, 2, 0), (2, 3, 3)), Roi((15, 12, 15), (2, 3, 3))) in result
    assert (Roi((3, 0, 0), (2, 1, 3)), Roi((13, 15, 15), (2, 1, 3))) in result
    assert (Roi((0, 0, 0), (2, 1, 3)), Roi((15, 15, 15), (2, 1, 3))) in result


def test_exact_boundary(buffer):
    roi = Roi((15, 15, 15), (5, 5, 5))
    result = buffer.wrap_logical_roi_into_buffer_rois(roi)
    assert len(result) == 1
    buffer_roi, logical_roi = result[0]
    assert buffer_roi == Roi((0, 0, 0), (5, 5, 5))
    assert logical_roi == roi
