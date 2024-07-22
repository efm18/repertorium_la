from utils.image_tools import Dimensions

SUPPORTED_OD_FORMATS = ['yolo', 'coco', 'pascal']

# COCO: [x, y, width, height]
# YOLO: [x_center, y_center, width, height]
# PASCAL: [x, y, x, y]

class BoundingBox:
    """
    Cropping of a bigger image
    Attributes:
        :from_x: Top left corner, x
        :from_y: Top left corner, y
        :to_x: Bottom right corner, x
        :to_y: Bottom right corner, y
    """

    def __init__(self, a: int, b: int, c: int, d: int, format: str = 'yolo'):
        if format not in SUPPORTED_OD_FORMATS:
            raise ValueError(f"Unsupported format '{format}', must be one of {SUPPORTED_OD_FORMATS}")
        
        self.current_format = format
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @staticmethod
    def from_MuRET(bbox: dict) -> 'BoundingBox':
        return BoundingBox(
            bbox['fromX'],
            bbox['fromY'],
            bbox['toX'],
            bbox['toY'],
            'pascal'
        )
    
    def resize(self, scale_factor: tuple):
        self.a = scale_factor[0] * self.a
        self.b = scale_factor[1] * self.b
        self.c = scale_factor[0] * self.c
        self.d = scale_factor[1] * self.d

    def to(self, format: str, img_dims: Dimensions | None = None) -> 'BoundingBox':
        return getattr(self, f'to_{format}')(img_dims)

    def to_yolo(self, img_dims: Dimensions | None = None) -> 'BoundingBox':
        if self.current_format == 'yolo':
            return self
        else:
            if img_dims is None:
                raise ValueError(f'Image dimensions are needed to convert to YOLO format from {self.current_format.upper()}.')
            
            img_width, img_height = img_dims
            if self.current_format == 'pascal':
                x_center = (self.a + self.c) / 2
                y_center = (self.b + self.d) / 2
                width = self.c - self.a
                height = self.d - self.b
            else: # COCO
                x_center = self.a + (self.c / 2) # x + (width / 2)
                y_center = self.b + (self.d / 2) # y + (height / 2)
                width = self.c
                height = self.d

            # Normalize values by image dimensions
            return BoundingBox(
                round(x_center / img_width, 5),
                round(y_center / img_height, 5),
                round(width / img_width, 5),
                round(height / img_height, 5),
                'yolo'
            )
        
    def to_coco(self, img_dims: Dimensions | None = None) -> 'BoundingBox':
        if self.current_format == 'coco':
            return self
        else:
            if self.current_format == 'yolo':
                if img_dims is None:
                    raise ValueError(f'Image dimensions are needed to convert to COCO format from YOLO.')
                
                img_width, img_height = img_dims
                width = self.c * img_width
                height = self.d * img_height
                x = self.a * img_width - (width / 2)
                y = self.b * img_height - (height / 2)
            else: # PASCAL
                x = self.a
                y = self.b
                width = self.c - self.a
                height = self.d - self.b
            
            return BoundingBox(x, y, width, height, 'coco')
        
    def to_pascal(self, img_dims: Dimensions | None = None) -> 'BoundingBox':
        if self.current_format == 'pascal':
            return self
        else:
            if self.current_format == 'yolo':
                if img_dims is None:
                    raise ValueError(f'Image dimensions are needed to convert to PASCAL format from YOLO.')
                
                img_width, img_height = img_dims
                width = self.c * img_width
                height = self.d * img_height
                x = self.a * img_width - (width / 2)
                y = self.b * img_height - (height / 2)
                x2 = x + width
                y2 = y + height
            else: # COCO
                x = self.a
                y = self.b
                x2 = self.a + self.c
                y2 = self.b + self.d
            
            return BoundingBox(x, y, x2, y2, 'pascal')

    def __str__(self) -> str:
        return f'{self.a} {self.b} {self.c} {self.d}'