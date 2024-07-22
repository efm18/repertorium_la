from .AgnosticSymbol import AgnosticSymbol
from ..YOLO.BoundigBox import BoundingBox

class Region:
    """
    It contains the representation of a region in a JSON file from MuRET
    Attributes:
        :symbols:           Division of the region into symbols
        :bounding_box:      Box delimiting the region
        :semantic_encoding: Semantic encoding string, possibly empty
        :type:       Staff, lyrics, etc...
    """

    bounding_box: BoundingBox
    semantic_encoding: str
    type: str
    symbols: list[AgnosticSymbol]

    def __init__(self, type: str, bounding_box: BoundingBox, semantic_encoding: str):
        self.bounding_box = bounding_box
        self.semantic_encoding = semantic_encoding
        self.type = type
        self.symbols = [] # initialized as empty

    def add_symbol(self, symbol: AgnosticSymbol):
        self.symbols.append(symbol)