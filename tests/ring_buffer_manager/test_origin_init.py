from itertools import product

from hypothesis import given
from hypothesis import strategies as st
from rbvr import Coordinate, RingBufferManager


def create_expected_from_origin(
    view_distance: Coordinate,
) -> set[tuple[Coordinate, Coordinate]]:
    nx, ny, nz = view_distance
    sx, sy, sz = view_distance * 2 + 1

    def expected_single(s, n):
        # the math gets funky with 0
        if n == 0:
            return {(0, 0)}
        return set(
            zip(
                range(s),
                (*range(n + 1), *range(-n, 0)),
            )
        )

    expected_x = expected_single(sx, nx)
    expected_y = expected_single(sy, ny)
    expected_z = expected_single(sz, nz)
    expected = {
        (Coordinate(a, b, c), Coordinate(x, y, z))
        for (a, x), (b, y), (c, z) in product(expected_x, expected_y, expected_z)
    }
    return expected


# we have to cap this because it takes too long
@given(
    st.integers(min_value=0, max_value=13),
    st.integers(min_value=0, max_value=13),
    st.integers(min_value=0, max_value=13),
)
def test_origin_init(x, y, z):
    view_distance = Coordinate(x, y, z)
    manager = RingBufferManager(view_distance)
    expected = create_expected_from_origin(view_distance)
    assert manager.initial_assignments == expected
