import numpy as np

from ._geometry import Coordinate, Roi

# an assigment should be used to do:
# ```py
# dst, src = assignment
# ring_buffer[dst] = data[src]
# ```
type CoordinateAssignment = tuple[Coordinate, Coordinate]
type RoiAssignment = tuple[Roi, Roi]


class RingBufferManager:
    # PyCharm tries to resolve references in the docstring examples
    # noinspection PyUnresolvedReferences
    """
    A manager for the state behind the ring buffer.

    This class represents the state of the ring buffer and returns the assignments
    that need to be performed to update the ring buffer when the camera moves. The
    class implementing the backing ring buffer should use the assignments returned
    from this class to update the ring buffer.

    Each Assignment is a tuple of two Coordinates to be used as such:

    >>> dst, src = assignment
    >>> ring_buffer[dst] = data[src]
    """

    view_distance: Coordinate
    size: Coordinate
    initial_assignments: set[CoordinateAssignment]

    def __init__(
        self,
        view_distance: Coordinate,
        initial_camera_chunk: Coordinate = Coordinate(0, 0, 0),
    ):
        # even though view distance isn't semantically a coordinate, we can still
        # use the coordinate class for convenience and math.
        self.view_distance = view_distance
        self._camera_chunk = initial_camera_chunk
        self._old_assignments = self._get_assignments_for_chunk(self._camera_chunk)
        # _old_assignments is internal, we only want to expose the initial assignments
        self.initial_assignments = self._old_assignments

    @property
    def size(self) -> Coordinate:
        """
        The full size of the ring buffer in chunks.
        """
        return Coordinate(2 * d + 1 for d in self.view_distance)

    def _mod_size(self, value: Coordinate) -> Coordinate:
        """
        Modulo the value with the size of the ring buffer.

        This is used to wrap around the ring buffer.
        """
        # this implementation converts a truncated modulo operation
        # into a Euclidean modulo operation. not sure if we need this?
        # if we require negative checks by the caller, we can just use
        # `value % self.size` directly.
        return ((value % self.size) + self.size) % self.size

    @property
    def _camera_index(self) -> Coordinate:
        return self._mod_size(self._camera_chunk)

    def _get_assignments_for_chunk(
        self, camera_chunk: Coordinate
    ) -> set[CoordinateAssignment]:
        """
        Get all the assignments needed for a given central camera chunk.

        Returns:
            A set of assignments to perform.
        """
        # This method is an AI translation of the original 1D version:
        # ```py
        # return [
        #     ((i % self.size + self.size) % self.size, i)
        #     for i in range(
        #         new_camera_chunk - self.view_distance,
        #         new_camera_chunk + self.view_distance + 1,
        #     )
        # ]
        # ```
        # Compute the ranges in each dimension
        # Parameters
        # Create range arrays per axis
        ranges = [
            np.arange(c - vd, c + vd + 1)
            for c, vd in zip(camera_chunk, self.view_distance)
        ]

        # 3D grid of original coordinates
        i, j, k = np.meshgrid(*ranges, indexing="ij")
        sx, sy, sz = self.size

        # Apply wrapping per axis
        w_i = (i % sx + sx) % sx
        w_j = (j % sy + sy) % sy
        w_k = (k % sz + sz) % sz

        # Stack into coordinate arrays
        wrapped_coords = np.stack((w_i, w_j, w_k), axis=-1)
        original_coords = np.stack((i, j, k), axis=-1)

        # Flatten and pair each coordinate
        return {
            (Coordinate(x), Coordinate(y))
            for x, y in zip(
                wrapped_coords.reshape(-1, 3),
                original_coords.reshape(-1, 3),
            )
        }

    def move_camera(self, new_camera_chunk: Coordinate) -> set[CoordinateAssignment]:
        """
        Move the camera to a new position and return the assignments needed to update the ring buffer.

        Returns:
            A set of assignments to perform.
        """
        # this should return
        #     k = new_camera_chunk - self._camera_chunk
        #     min(k, self.size)
        # if k >= self.size, should just do self._replace_with_camera(new_camera_chunk)
        new_assignments = self._get_assignments_for_chunk(new_camera_chunk)
        difference = new_assignments - self._old_assignments
        self._old_assignments = new_assignments
        self._camera_chunk = new_camera_chunk
        return difference


class WorldCoordinateRingBufferManager(RingBufferManager):
    """
    A special ring buffer that:

    * knows about chunk sizes
    * returns Rois instead of Coordinates
    * can filter out negative coordinates
    * can filter out coordinates that are outside the volume dimensions
    """

    initial_assignments: set[RoiAssignment]

    def __init__(
        self,
        chunk_dimensions: Coordinate,
        view_distance: Coordinate,
        volume_dimensions: Coordinate | None = None,
        initial_camera_coordinate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Don't use this. Use `init_with_ops` instead.
        """
        self.chunk_dimensions = chunk_dimensions
        self.volume_dimensions = volume_dimensions
        initial_camera_chunk = self._camera_coordinate_to_chunk(
            initial_camera_coordinate
        )
        super().__init__(view_distance, initial_camera_chunk)
        # initial assignments is a coordinate list at first
        # this is where we convert it to a Roi list
        # noinspection PyTypeChecker
        self.initial_assignments = self._coordinate_set_to_roi_set(
            self.initial_assignments
        )

    @property
    def real_size(self):
        return self.size * self.chunk_dimensions

    def _camera_coordinate_to_chunk(
        self, camera_coordinate: tuple[float, float, float]
    ) -> Coordinate:
        return Coordinate(
            x // y for x, y in zip(camera_coordinate, self.chunk_dimensions)
        )

    def _coordinate_to_roi(
        self, assignment: CoordinateAssignment
    ) -> RoiAssignment | None:
        dst, src = assignment
        dst *= self.chunk_dimensions
        src *= self.chunk_dimensions
        # we don't want any negatives
        if any(d < 0 for d in src) or any(d < 0 for d in dst):
            return None
        if self.volume_dimensions is not None and any(
            i >= d for i, d in zip(src, self.volume_dimensions)
        ):
            return None
        dst = Roi(offset=dst, shape=self.chunk_dimensions)
        src = Roi(offset=src, shape=self.chunk_dimensions)
        return dst, src

    def _coordinate_set_to_roi_set(
        self, assignments: set[CoordinateAssignment]
    ) -> set[RoiAssignment]:
        valid_assignments = set()
        for coordinate_assignment in assignments:
            roi = self._coordinate_to_roi(coordinate_assignment)
            if roi is None:
                continue
            valid_assignments.add(roi)
        return valid_assignments

    def move_camera(
        self, new_camera_coordinate: tuple[float, float, float]
    ) -> set[RoiAssignment]:
        new_camera_chunk = self._camera_coordinate_to_chunk(new_camera_coordinate)
        # if the camera chunk hasn't changed, we don't need to do anything
        if new_camera_chunk == self._camera_chunk:
            return set()

        assignments = super().move_camera(new_camera_chunk)

        return self._coordinate_set_to_roi_set(assignments)
