from .Region import Region
from ..YOLO.BoundigBox import BoundingBox

class Page:
    """
    It contains the representation of a page in a JSON file from MuRET
    Attributes:
        :regions: Division of the page into regions
        :bounding_box: Box delimiting the page
    """

    def __init__(self, bounding_box: BoundingBox):
        self.regions = []
        self.bounding_box = bounding_box

    def add_region(self, region: Region):
        self.regions.append(region)