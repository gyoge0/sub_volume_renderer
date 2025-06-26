import numpy as np
import pytest
from funlib.geometry import Coordinate
from rbvr._wrapping_buffer import WrappingBuffer


@pytest.fixture
def chunk_size():
    return Coordinate((4, 4, 4))


@pytest.fixture
def buffer_chunks():
    return Coordinate((5, 5, 5))


@pytest.fixture
def backing_data():
    arr = np.arange(40 * 40 * 40, dtype=np.uint16).reshape((40, 40, 40))
    return arr


@pytest.fixture
def buffer(backing_data, chunk_size, buffer_chunks):
    return WrappingBuffer(backing_data, buffer_chunks, chunk_size)
