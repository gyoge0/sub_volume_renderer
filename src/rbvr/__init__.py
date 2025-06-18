from ._types import Float32Array
from ._wobject import GlobalSparseVolume

# we need to import the shader module so that the shader is registered
# it's not imported anywhere else
from ._shader import GlobalSparseVolumeShader  # noqa: F401 # isort: skip

__all__ = [
    "GlobalSparseVolume",
    "Float32Array",
]
