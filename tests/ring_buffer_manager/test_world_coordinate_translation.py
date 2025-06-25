from hypothesis import given
from hypothesis import strategies as st
from rbvr import Coordinate, Roi, WorldCoordinateRingBufferManager


# this test is just the implementation... do we really need it?
@given(
    st.tuples(
        st.integers(min_value=1),
        st.integers(min_value=1),
        st.integers(min_value=1),
    ),
    st.tuples(
        st.integers(min_value=0),
        st.integers(min_value=0),
        st.integers(min_value=0),
    ),
    st.tuples(
        st.integers(min_value=0),
        st.integers(min_value=0),
        st.integers(min_value=0),
    ),
)
def test_coordinate_to_roi(chunk_dimensions, src, dst):
    chunk_dimensions = Coordinate(chunk_dimensions)
    src = Coordinate(src)
    dst = Coordinate(dst)

    manager = WorldCoordinateRingBufferManager(
        chunk_dimensions,
        # view distance doesn't matter for this test
        view_distance=Coordinate(0, 0, 0),
    )

    expected_src = Roi(offset=src * chunk_dimensions, shape=chunk_dimensions)
    expected_dst = Roi(offset=dst * chunk_dimensions, shape=chunk_dimensions)

    roi = manager._coordinate_to_roi((dst, src))
    assert roi == (expected_dst, expected_src)
