from typing import List
from numpy import ndarray

from utils.image_tools import Dimensions
from .Object import Object

class Image:
    """
    It contains the representation of an image information in YOLO format
    Attributes:
    """

    file_name: str
    image_array: ndarray
    objects: List[Object]

    def __init__(self, file_name: str, image_array: ndarray, objects: List[Object], img_dims: Dimensions):
        self.file_name = file_name
        self.image_array = image_array
        self.objects = objects
        self.img_dims = img_dims # Original image dimensions
        resized_dims = Dimensions(self.image_array.shape[1], self.image_array.shape[0])

        # Preprocess objects
        for obj in self.objects:
            obj.preprocess('yolo', img_dims, resized_dims)

    def __str__(self) -> str:
        return '\n'.join([str(obj) for obj in self.objects])