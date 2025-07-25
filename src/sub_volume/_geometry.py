# since we reexport Roi, we might as well reexport Coordinate from here as well.
# we don't have to change imports if we end up modifying the Coordinate API like how
# we did with Roi. the reason we import Coordinate here is because that way our internal
# modules can import from _roi without having to import funlib.geometry directly.
from warnings import deprecated

from funlib.geometry import Coordinate
from funlib.geometry import Roi as OriginalRoi


@deprecated("Will be removed when the new ring buffer API is implemented.")
class Roi(OriginalRoi):
    """A ROI that can be hashed by the offset and shape."""

    def __hash__(self):
        return hash((self.offset, self.shape))


__all__ = ["Roi", "Coordinate"]
