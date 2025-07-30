"""
A Pygfx-based volume renderer for large electron microscopy datasets.

This package provides volume rendering with multi-scale data support using
3D ring buffers and level-of-detail rendering for efficient handling of
large datasets.
"""

from ._material import SubVolumeMaterial
from ._wobject import SubVolume
from ._wrapping_buffer import WrappingBuffer

# we need to import the shader module so that the shader is registered
# it's not imported anywhere else
from ._shader import SubVolumeShader  # noqa: F401 # isort: skip

__all__ = [
    "SubVolume",
    "SubVolumeMaterial",
    "WrappingBuffer",
]
