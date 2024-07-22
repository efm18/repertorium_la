from ..YOLO.BoundigBox import BoundingBox

class AgnosticSymbol:
    """
    It contains the representation of an agnostic symbol in a JSON file from MuRET
    Attributes:
        :agnostic_symbol_type:  The label of the symbol
        :position_in_staff: Vertical position of the symbol in the staff
        :bounding_box: Box delimiting the page, possibly empty
        :approximate_x: When bounding_box is unknown, it is horizontally positioned with this x
    """

    agnostic_symbol_type: str
    position_in_staff: str
    bounding_box: BoundingBox
    approximate_x: int

    def __init__(self, agnostic_symbol_type: str, position_in_staff: str, bounding_box: BoundingBox,
                 approximate_x: int):
        self.agnostic_symbol_type = agnostic_symbol_type
        self.position_in_staff = position_in_staff
        self.bounding_box = bounding_box
        self.approximate_x = approximate_x