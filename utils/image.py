import os
import time
import PIL
import socket
import logging
import requests
from numpy import ndarray
from hashlib import sha256
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import urllib.request
from urllib.error import HTTPError, URLError

from utils.sync import SynchronizedDiskAccess
from utils.image_tools import Dimensions, ImageTranscoder

T = TypeVar('T')

class AbstractImage(Generic[T], ABC):
    """Abstraction of what an image is. It can be a local file, a remote file and is represented on an internal format.

    Attributes:
        representation          Internal image representation
        url                     Unique name
        cache_files_folder      Path where local files are stored. If specified
        cropped_files_folder    When cache_files_folder is provided, this subfolder is created to store cropped images
    """

    representation: T
    url: str
    cache_files_folder: str
    cropped_files_folder: str

    def __init__(self, representation: T, cache_files_folder: str = None):
        if representation is None:
            raise ValueError('Representation cannot be None')
        self.representation = representation

        # Check ir url exists in the object
        if not hasattr(representation, 'url'):
            raise AssertionError('Representation must have a URL')
        self.url = representation.url

        self.cache_files_folder = cache_files_folder
        if self.cache_files_folder is not None:
            self.cropped_files_folder = os.path.join(self.cache_files_folder, 'cropped')
            os.makedirs(self.cropped_files_folder, exist_ok=True)
        else:
            self.cropped_files_folder = None

    @abstractmethod
    def _get_content(self) -> PIL.Image:
        """
        It retrieves the PIL.Image object as content
            :return:
        """
        ...

    def _get_local_filename(self, char_limit: str = 13) -> str:
        """
        It converts the url into a hash string
            :param char_limit: The length of the hash string
            :return: Encoded local file name
        """
        hash_code = sha256(self.url.encode()).hexdigest()[:char_limit]
        return hash_code

    def __get_local_cropped_absolute_path(self, transcoder: ImageTranscoder, from_x: int, from_y: int, to_x: int,
                                          to_y: int) -> str:
        """
        It adds the cropping coordinates to the path
            :param transcoder: Its ID will be used in the file name
            :param from_x: Left top corner x coordinate
            :param from_y: Left top corner y coordinate
            :param to_x: Bottom right x coordinate
            :param to_y: Bottom right y coordinate
            :return: Full path
        """
        return os.path.join(self.__get_local_cropped_file_folder(transcoder),
                            f'{from_x}_{from_y}_{to_x}_{to_y}.{transcoder.file_extension()}')

    def __get_local_cropped_file_folder(self, transcoder: ImageTranscoder) -> str:
        """
        It builds a path containing the transcoder ID
            :param transcoder: Image transcoder
            :return: Fill path of the images encoded using the provided transcoder
        """
        return os.path.join(self.cropped_files_folder, transcoder.id(), self._get_local_filename())

    def load_cropped(self, transcoder: ImageTranscoder, from_x: int, from_y: int, to_x: int, to_y: int) -> ndarray:
        """
        It delegates the encoding of the image to a specialized class
            :param transcoder: The transcoder to convert the image
            :param from_x: Top left x coordinate
            :param from_y: Top left y coordinate
            :param to_x: Bottom right x coordinate
            :param to_y: Bottom right y coordinate
            :return: 2d ndarray
        """
        if self.cache_files_folder is None:
            content = self._get_content()
            result = transcoder.encode_cropped(self._get_content(), from_x, from_y, to_x, to_y)
            return result
        else:
            cached_file = self.__get_local_cropped_absolute_path(transcoder, from_x, from_y, to_x, to_y)

            if not os.path.exists(cached_file):
                content = self._get_content()
                cropped = transcoder.encode_cropped(content, from_x, from_y, to_x, to_y)
                logging.debug(f'Saving cropped image to {cached_file}')
                cached_file_folder = self.__get_local_cropped_file_folder(transcoder)

                SynchronizedDiskAccess.create_if_not_exists(cached_file_folder)
                transcoder.save(cached_file, cropped)
                return cropped
            else:
                logging.debug(f'Loading cropped image from {cached_file}')
                cropped = transcoder.load(cached_file)
                return cropped

    def load_image(self, transcoder: ImageTranscoder, *args, **kwargs) -> ndarray:
        """
        It loads the encoded image using the given transcoder
            :param transcoder: The transforming object
            :return: 2d ndarray
        """
        content = self._get_content()
        result = transcoder.encode(content, *args, **kwargs)
        return result

    def dimensions(self) -> Dimensions:
        """
        It retrieves the dimensions of the image
            :return: Dimensions object
        """
        content = self._get_content() # content is a PIL Image
        width = content.size[0]
        height = content.size[1]
        return Dimensions(width, height)
    
class RemoteImage(AbstractImage[T]):
    """Image located at a URL, e.g., an IIIF server. If so configured, it is saved locally in a cache folder. In
    that case when instantiated, it is looked into the cache folder before downloading it again.

    Attributes:
        disk_file_name      Full path of the local file
        file_name           Local file name
        images_cache        Folder where the images are stored
    """

    disk_file_name: str
    file_name: str
    images_cache: str

    def __init__(self, *args):
        AbstractImage.__init__(self, *args)

        if self.cache_files_folder is None:
            self.disk_file_name = None
            self.images_cache = None
        else:
            self.images_cache = os.path.join(self.cache_files_folder, 'images')
            self.file_name = self._get_local_filename()
            self.disk_file_name = os.path.join(self.images_cache, self.file_name)

            if not os.path.exists(self.images_cache):
                os.makedirs(self.images_cache)

            if not os.path.exists(self.disk_file_name):
                content = self.__download_url_image()
                if content:
                    with open(self.disk_file_name, 'wb') as f:
                        logging.debug(f'Saving file from URL {self.url} in {self.disk_file_name} ')
                        f.write(content)
                    f.close()
                else:
                    raise Exception('Cannot retrieve image')
            else:
                logging.debug(f'Using cached file for URL {self.url} from {self.disk_file_name}')

    def __download_url_image(self):
        """
        :return: It returns an object which can work as a context manager. (see urllib.request.urlopen)
        """
        logging.debug(f'Connecting to URL {self.url}')
        try:
            url_image = urllib.request.urlopen(self.url, timeout=10)
            
            # Avoiding BNF Gallica server to block the request
            if self.url.find('gallica.bnf.fr') != -1:
                time.sleep(10)
            
            logging.debug(f'Reading from {self.url}')

            content = url_image.read()
            logging.debug(f'Successful download from {self.url}')
            return content
        except HTTPError as error:
            logging.error('Data not retrieved because %s\nURL: %s', error, self.url)
        except URLError as error:
            if isinstance(error.reason, socket.timeout):
                logging.error('Socket timed out - url %s', self.url)
            else:
                logging.error('Some other error happened while downloading url %s', self.url);
        except Exception as e:
            logging.error('Some other error happened while downloading url %s', self.url);


    def _get_content(self) -> PIL.Image:
        """
        [OVERRIDE] It retrieves the image from the disk or from the url
            :return: A PIL.Image object, that will be instantiated as a derived class such as JpegImageFile
        """
        if self.disk_file_name is None:
            logging.debug(f'Downloading file from URL {self.url}')
            response = requests.get(self.url, stream=True)
            response.raw.decode_content = True
            
            with PIL.Image.open(response.raw) as img:
                img.load()  # Load image while the stream is open
                return img.copy()  # Return a copy of the image
        else:
            with PIL.Image.open(self.disk_file_name) as img:
                return img.copy()  # Return a copy of the image
        
class LocalImage(AbstractImage[T]):
    """Image located at the disk. It uses the PIL image tool format
    Attributes:
        files_folder    Where files are located
    """

    def __init__(self, files_folder: str, relative_file_name: str):
        AbstractImage.__init__(relative_file_name, files_folder)
        self.files_folder = files_folder

    def _get_content(self) -> PIL.Image:
        disk_file_name = os.path.join(self.files_folder, self.relative_file_name)
        return PIL.Image.open(disk_file_name)
