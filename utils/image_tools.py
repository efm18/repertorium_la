import numpy
import logging
from PIL import Image
from numpy import ndarray
from PIL.Image import Resampling
from abc import ABC, abstractmethod

class Transcoder(ABC):
    pass

class ImageTranscoder(Transcoder, ABC):
    """
    It retrieves an image in a given encoding, number of channels, size, etc.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, pil_image: Image, width: int = None, height: int = None) -> ndarray:
        """
        It transforms the image into a 2d ndarray, resizing it if necessary.
            :param pil_image: Input image
            :param width: Target width
            :param height: Target height
            :return: 2d ndarray
        """
        ...

    @abstractmethod
    def encode_cropped(self, pil_image: Image, from_x: int, from_y: int, to_x: int, to_y: int) -> ndarray:
        """
        It transforms the input image into a 2d ndarray, cropping the image first to the specified region.
            :param pil_image: Image to be encoded
            :param from_x: Top left corner x component
            :param from_y: Top left corner y component
            :param to_x: Bottom right x component
            :param to_y: Bottom right y component
            :return: 2d ndarray
        """
        ...

    @abstractmethod
    def save(self, cached_file: str, encoding: ndarray):
        """
        It saves the image into a local file as .npy
            :param cached_file: Full path of the local file
            :param encoding: 2d ndarray
            :return: None
        """
        ...

    @abstractmethod
    def save_as_png(self, cached_file: str, encoding: ndarray):
        """
        It saves the image into a local file as png
            :param cached_file: Full path of the local file
            :param encoding: 2d ndarray
            :return: None
        """
        ...

    @abstractmethod
    def load(self, cached_file: str) -> ndarray:
        """
        It loads a 2d ndarray from disk
            :param cached_file: Full path of the image
            :return: 2d ndarray
        """
        ...

    @abstractmethod
    def file_extension(self) -> str:
        """
        It returns the file extension used by the encoding
            :return: The extension preceded with a dot, e.g. .npy
        """
        ...

    @abstractmethod
    def id(self) -> str:
        """
        Used for distinguishing transcoders to save cropped / processed regions - use only path compliant characters
            :return: A unique ID
        """
        ...


class NumpyGrayscaleImageTranscoder(ImageTranscoder):
    """
    It encodes the image into a 2d ndarray containing values from 0 to 255 representing a grayscale image
    """

    def __init__(self):
        ImageTranscoder.__init__(self)

    def file_extension(self) -> str:
        """
        [OVERRIDE] It returns the file extension used by the encoding
            :return: The extension preceded with a dot, e.g. .npy
        """
        return 'npy'

    def encode(self, pil_image: Image, width: int | None = None, height: int | None = None) -> ndarray:
        """
        [OVERRIDE] It transforms the image into a 2d ndarray, resizing it if necessary.
            :param pil_image: Input image
            :param width: Target width
            :param height: Target height
            :return: 2d ndarray
        """
        logging.debug('Encoding image')
        resized_img = pil_image

        if width is None:
            width = pil_image.size[0]
        if height is None:
            height = pil_image.size[1]

        # Ensure width and height is positive
        if width <= 0 and height <= 0:
            raise ValueError('Width and height must be positive')

        if width > pil_image.size[0] or height > pil_image.size[1]:
            logging.warning(f'Target dimensions ({width}, {height}) are larger than the original image ({pil_image.size[0]}, {pil_image.size[1]})')

        # Resize image if necessary
        if pil_image.size[0] != width or pil_image.size[1] != height:
            logging.debug(f'Resizing image to w={width}, h={height}')
            resized_img = resized_img.resize([width, height], Resampling.NEAREST)

        # Convert image to grayscale
        logging.debug('Converting image to grayscale')
        grayscale = resized_img.convert("L")
        return numpy.asarray(grayscale)

    def encode_cropped(self, pil_image: Image, from_x: int, from_y: int, to_x: int, to_y: int) -> ndarray:
        """
        [OVERRIDE] It transforms the input image into a 2d ndarray, cropping the image first to the specified region.
            :param pil_image: Image to be encoded
            :param from_x: Top left corner x component
            :param from_y: Top left corner y component
            :param to_x: Bottom right x component
            :param to_y: Bottom right y component
            :return: 2d ndarray
        """

        if from_x >= to_x or from_y >= to_y:
            raise ValueError('Resulting crop must have positive dimensions')
        
        if from_x < 0 or from_y < 0 or to_x < 0 or to_y < 0:
            raise ValueError('Crop coordinates must be positive')

        if from_x >= pil_image.size[0] or from_y >= pil_image.size[1] or to_x >= pil_image.size[0] or to_y >= pil_image.size[1]:
            raise ValueError('Coordinates must not exceed the image dimensions')
        
        logging.debug(f'Encoding cropped image ({from_x}, {from_y}) to ({to_x}, {to_y})')
        cropped_img = pil_image.crop((from_x, from_y, to_x, to_y))  # double parenthesis is not a mistake
        result = self.encode(cropped_img)
        return result

    def save(self, cached_file: str, encoding: ndarray):
        """
        [OVERRIDE] It saves the image into a local file as .npy
            :param cached_file: Full path of the local file
            :param encoding: 2d ndarray
            :return: None
        """
        numpy.save(cached_file, encoding, False)

    def save_as_png(self, filename_without_extension: str, encoding: ndarray):
        """
        [OVERRIDE] It saves the image into a local file as png
            :param filename_without_extension: Full path of the local file without the extension
            :param encoding: 2d ndarray
            :return: None
        """
        im = Image.fromarray(encoding)
        im.save(filename_without_extension + '.png')

    def load(self, cached_file: str) -> ndarray:
        """
        [OVERRIDE] It loads a 2d ndarray from disk
            :param cached_file: Full path of the image
            :return: 2d ndarray
        """
        result = numpy.load(cached_file, None, False)
        return result

    def id(self) -> str:
        """
        [OVERRIDE] Used for distinguishing transcoders to save cropped / processed regions - use only path compliant characters
            :return: A unique ID
        """
        return 'numpy_gray'
        # return f'numpy_gray_w{self.target_width}_h{self.target_height}'

########################################################################################################################

class Dimensions:
    """
    Image dimensions
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __iter__(self):
        yield self.width
        yield self.height

    def __str__(self) -> str:
        return f'[w={self.width}, h={self.height}]'