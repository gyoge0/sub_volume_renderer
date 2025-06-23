import numpy as np
import numpy.typing as npt
import pygfx as gfx
from pygfx import WorldObject
from pygfx.utils.bounds import Bounds


class GlobalSparseVolumeMaterial(gfx.VolumeMipMaterial):
    pass


class GlobalSparseVolume(gfx.Volume):
    uniform_type = dict(
        WorldObject.uniform_type,
        chunk_dimensions="3xf4",
        ring_buffer_dimensions_in_chunks="3xf4",
        volume_dimensions="3xf4",
    )
    material: GlobalSparseVolumeMaterial

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        chunk_dimensions: tuple[int, int, int],
        ring_buffer_n: tuple[int, int, int] = (2, 2, 2),
    ):
        assert len(data.shape) == 3, "Data must be a 3D ndarray"

        self.volume_dimensions = data.shape
        self.chunk_dimensions = chunk_dimensions
        self.volume_dimensions_in_chunks = tuple(
            d // c for d, c in zip(self.volume_dimensions, self.chunk_dimensions)
        )

        self.ring_buffer_dimensions_in_chunks = (
            2 * ring_buffer_n[0] + 1,
            2 * ring_buffer_n[1] + 1,
            2 * ring_buffer_n[2] + 1,
        )
        self._ring_buffer_shape = (
            self.ring_buffer_dimensions_in_chunks[0] * self.chunk_dimensions[0],
            self.ring_buffer_dimensions_in_chunks[1] * self.chunk_dimensions[1],
            self.ring_buffer_dimensions_in_chunks[2] * self.chunk_dimensions[2],
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
            material=GlobalSparseVolumeMaterial(),
        )

        # we need to call super before we can set the uniform buffer
        self.uniform_buffer.data["volume_dimensions"] = np.array(
            self.volume_dimensions, dtype=np.float32
        )
        self.uniform_buffer.data["chunk_dimensions"] = np.array(
            self.chunk_dimensions, dtype=np.float32
        )
        self.uniform_buffer.data["ring_buffer_dimensions_in_chunks"] = np.array(
            self.ring_buffer_dimensions_in_chunks, dtype=np.float32
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
