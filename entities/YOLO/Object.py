from utils.image_tools import Dimensions
from .BoundigBox import BoundingBox

class Object:
    """
    This class represents an object to be recognized by the object detection 
    algorithm. It is a simple class that holds the object's label and the
    bounding box. Also a string representation of the object is provided.

    Attributes:
        label: Label of the object
        bounding_box: Bounding box of the object
    """

    def __init__(self, label: int, bounding_box: BoundingBox) -> None:
        self.label = label
        self.bounding_box = bounding_box

    def preprocess(self, format: str, img_dims: Dimensions, resized_dims: Dimensions) -> 'Object':
        scale_factor = (resized_dims.width / img_dims.width, resized_dims.height / img_dims.height)
        self.bounding_box.resize(scale_factor)
        self.bounding_box = self.bounding_box.to(format, resized_dims)

    def __str__(self) -> str:
        return f'{self.label} {self.bounding_box}'