import numpy as np
import pytest


@pytest.fixture
def chunk_dimensions():
    return 5, 5, 5


@pytest.fixture
def data_shape():
    return 25, 25, 25


@pytest.fixture
def increasing_data(data_shape):
    x, y, z = data_shape
    return np.arange(x * y * z).reshape(data_shape).astype(np.float32)
