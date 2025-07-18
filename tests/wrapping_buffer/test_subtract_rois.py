from itertools import combinations

from funlib.geometry import Roi
from hypothesis import given
from hypothesis import strategies as st

# noinspection PyProtectedMember
from sub_volume._wrapping_buffer import subtract_rois


def test_no_overlap():
    a = Roi((0, 0, 0), (2, 2, 2))
    b = Roi((4, 4, 4), (2, 2, 2))
    result = subtract_rois(a, b)
    assert result == [a]


def test_full_containment():
    a = Roi((0, 0, 0), (4, 4, 4))
    b = Roi((0, 0, 0), (4, 4, 4))
    result = subtract_rois(a, b)
    assert result == []


def test_partial_overlap_lower():
    a = Roi((0, 0, 0), (4, 4, 4))
    b = Roi((0, 0, 0), (2, 4, 4))
    result = subtract_rois(a, b)
    assert len(result) == 1
    assert result[0].offset == (2, 0, 0)
    assert result[0].shape == (2, 4, 4)


def test_partial_overlap_upper():
    a = Roi((0, 0, 0), (4, 4, 4))
    b = Roi((2, 0, 0), (2, 4, 4))
    result = subtract_rois(a, b)
    assert len(result) == 1
    assert result[0].offset == (0, 0, 0)
    assert result[0].shape == (2, 4, 4)


def test_partial_overlap_middle():
    a = Roi((0, 0, 0), (6, 4, 4))
    b = Roi((2, 0, 0), (2, 4, 4))
    result = subtract_rois(a, b)
    assert len(result) == 2
    offsets = [r.offset for r in result]
    shapes = [r.shape for r in result]
    assert (0, 0, 0) in offsets
    assert (4, 0, 0) in offsets
    assert (2, 4, 4) in shapes
    assert (2, 4, 4) in shapes


def test_touching_but_not_overlapping():
    a = Roi((0, 0, 0), (2, 2, 2))
    b = Roi((2, 0, 0), (2, 2, 2))
    result = subtract_rois(a, b)
    assert result == [a]


def test_empty_roi():
    a = Roi((0, 0, 0), (0, 0, 0))
    b = Roi((0, 0, 0), (2, 2, 2))
    result = subtract_rois(a, b)
    assert result == []


def test_overlap_with_empty_b():
    a = Roi((0, 0, 0), (2, 2, 2))
    b = Roi((0, 0, 0), (0, 0, 0))
    result = subtract_rois(a, b)
    assert result == [a]


def test_overlap_2_axes():
    a = Roi((0, 0, 0), (4, 4, 4))
    b = Roi((2, 2, 0), (2, 2, 4))
    result = subtract_rois(a, b)

    # At most 2 * 2 = 4 slabs should be returned
    assert len(result) <= 4
    for r in result:
        assert not r.intersects(b)
        assert a.contains(r)
    assert sum(x.size for x in result) == a.size - a.intersect(b).size
    assert all(not a.intersects(b) for a, b in combinations(result, 2))


def test_overlap_3_axes():
    a = Roi((0, 0, 0), (4, 4, 4))
    b = Roi((2, 2, 2), (2, 2, 2))
    result = subtract_rois(a, b)

    # At most 2 * 3 = 6 slabs should be returned
    assert len(result) <= 6
    for r in result:
        assert not r.intersects(b)
        assert a.contains(r)
    assert sum(x.size for x in result) == a.size - a.intersect(b).size
    assert all(not a.intersects(b) for a, b in combinations(result, 2))


@st.composite
def two_random_rois(draw):
    length = draw(st.integers(min_value=1, max_value=10))
    # we cap at 1e10 to avoid issues with unrealistically large numbers
    # in case 1e10 becomes realistic, we can just up it
    nums = st.integers(min_value=0)
    lists = st.lists(nums, min_size=length, max_size=length)

    offset_a = draw(lists)
    shape_a = draw(lists)
    offset_b = draw(lists)
    shape_b = draw(lists)
    return (
        Roi(tuple(offset_a), tuple(shape_a)),
        Roi(tuple(offset_b), tuple(shape_b)),
        length,
    )


@given(two_random_rois())
def test_property_no_overlap_with_b(data):
    a, b, length = data
    result = subtract_rois(a, b)

    assert len(result) <= 2 * length
    for r in result:
        assert not r.intersects(b)
        assert a.contains(r)

    actual_size = sum(x.size for x in result)
    expected_size = a.size - a.intersect(b).size
    assert actual_size == expected_size

    assert all(not a.intersects(b) for a, b in combinations(result, 2))
