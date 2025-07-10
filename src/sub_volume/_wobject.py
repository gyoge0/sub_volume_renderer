import numpy as np
import numpy.typing as npt
import pygfx as gfx
from pygfx import WorldObject
from pygfx.utils.bounds import Bounds

from ._geometry import Coordinate


class SubVolumeMaterial(gfx.VolumeMipMaterial):
    pass


class SubVolume(gfx.Volume):
    uniform_type = dict(
        WorldObject.uniform_type,
        chunk_dimensions="3xf4",
        ring_buffer_dimensions_in_chunks="3xf4",
        volume_dimensions="3xf4",
        ring_buffer_n="3xf4",
    )
    material: SubVolumeMaterial

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        chunk_dimensions: Coordinate,
        ring_buffer_n: Coordinate = Coordinate(2, 2, 2),
    ):
        assert len(data.shape) == 3, "Data must be a 3D ndarray"

        self.volume_dimensions = data.shape
        self.chunk_dimensions = chunk_dimensions
        self.volume_dimensions_in_chunks = Coordinate(
            d // c for d, c in zip(self.volume_dimensions, self.chunk_dimensions)
        )

        self.ring_buffer_n = ring_buffer_n
        self.ring_buffer_dimensions_in_chunks = ring_buffer_n * 2 + 1
        self._ring_buffer_shape = (
            self.ring_buffer_dimensions_in_chunks * self.chunk_dimensions
        )
        # noinspection PyTypeChecker
        self.ring_buffer_texture = gfx.Texture(
            data=np.ones(self._ring_buffer_shape, dtype=np.float32),
            dim=3,
        )

        # we make a dummy geometry here since we override self._get_bounds_from_geometry later, which is what matters
        # more. debatable on if we should do a proper geometry or not.
        geometry = gfx.box_geometry(*self.volume_dimensions)
        super().__init__(
            geometry=geometry,
            material=SubVolumeMaterial(),
        )

        # indexing in the shader is done Fortran style (z, y, x), but these dimensions
        # all assume numpy/C style indexing (x, y, z). we pass the dimensions in Fortran
        # style to the shader so the shader completely operates in Fortran style.
        self.uniform_buffer.data["volume_dimensions"] = np.array(
            self.volume_dimensions[::-1], dtype=np.float32
        )
        self.uniform_buffer.data["chunk_dimensions"] = np.array(
            self.chunk_dimensions[::-1], dtype=np.float32
        )
        self.uniform_buffer.data["ring_buffer_dimensions_in_chunks"] = np.array(
            self.ring_buffer_dimensions_in_chunks[::-1], dtype=np.float32
        )
        self.uniform_buffer.data["ring_buffer_n"] = np.array(
            self.ring_buffer_n[::-1], dtype=np.float32
        )

    def _get_bounds_from_geometry(self):
        if self._bounds_geometry is not None:
            return self._bounds_geometry
        # account for multi-channel image data
        grid_shape = tuple(reversed(self.volume_dimensions))
        # create aabb in index/data space
        aabb = np.array([np.zeros_like(grid_shape), grid_shape[::-1]], dtype="f8")
        # convert to local image space by aligning
        # center of voxel index (0, 0, 0) with origin (0, 0, 0)
        aabb -= 0.5
        # ensure coordinates are 3D
        # NOTE: important we do this last, we don't want to apply
        # the -0.5 offset to the z-coordinate of 2D images
        if aabb.shape[1] == 2:
            aabb = np.hstack([aabb, [[0], [0]]])
        self._bounds_geometry = Bounds(aabb, None)
        return self._bounds_geometry
