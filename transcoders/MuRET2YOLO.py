import os
import json
import logging
import threading
from typing import List
import concurrent.futures
from types import SimpleNamespace

from utils.image import RemoteImage
from utils.data import PartitionManager
from entities.YOLO.Image import Image as ImageYOLO
from entities.YOLO.Object import Object
from transcoders.Transcoder import EncodingTranscoder
from entities.MuRET import Dictionary, Package, Region, AgnosticSymbol, Image as ImageMuRET, ObjectsToDetectKind as otdk
from utils.image_tools import ImageTranscoder

class MuRET2YOLO(EncodingTranscoder):
    """
    It generates a YOLO compatible training dataset from a SymbolsDataset
    Attributes:
        :images:        dict, key = MuRET image, value = remote image
        :lock:          For concurrent access
    """

    def __init__(self, 
                 input_muret_training_set_folder: str, 
                 splits: SimpleNamespace,
                 cache_folder: str,
                 image_transcoder: ImageTranscoder,
                 *args, 
                 resize: tuple | int | None = None, 
                 **kwargs):
        logging.info(f'Importing MuRET training package from {input_muret_training_set_folder}')
        self.training_package = Package(input_muret_training_set_folder)
        self.image_transcoder = image_transcoder
        self.original_images = self._load_images(cache_folder)
        
        if isinstance(resize, tuple):
            if len(resize) != 2:
                raise ValueError(f"If 'resize' is a tuple, it has to have exactly 2 elements not {len(resize)}")
        else:
            resize = (resize, resize)
        self.resize = resize

        super().__init__(*args, **kwargs)

        for partition in ['train', 'validation', 'test']:
            if not hasattr(splits, partition):
                raise ValueError(f"No values id define for partition '{partition}' in splits")
        self.splits = splits

    def _load_images(self, cache_folder: str) -> List[ImageMuRET]:
        """
        It loads the images from the cache folder
            :param cache_folder: Cache folder where the images are stored
            :return: List of images
        """
        logging.info(f'Downloading images from URLs into cache folder {cache_folder}')
        lock = threading.Lock()
        images = []
        
        def load_single_image(image: ImageMuRET) -> ImageMuRET:
            lock.acquire()  # to avoid checking for the same file
            try:
                remote_image = RemoteImage[ImageMuRET](image, cache_folder)
            finally:
                lock.release()
            return remote_image
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for image in self.training_package.training_set_images:
                future_original_image = executor.submit(load_single_image, image)
                futures.append(future_original_image)

            for future in concurrent.futures.as_completed(futures):
                try:
                    image = future.result()
                    images.append(image)
                except Exception as e:
                    print(f'Cannot retrieve image {e}')

        return images

    def _create_output_structure(self, output_folder: str):
        for split in ['train', 'validation', 'test']:
            for kind in ['images', 'labels']:
                path = os.path.join(output_folder, kind, split)
                os.makedirs(path, exist_ok=True)
                setattr(self, f'{kind}.{split}', path)

    def __generate_dataYAML(self, output_folder: str):
        """
        It generates the data.yaml file
            :param output_folder: Output folder where the dataset will be saved
        """
        dictionary: Dictionary = self.training_package.region_types if self.isStaffWise else self.training_package.agnostic_symbol_types
        yaml_abspath = os.path.join(output_folder, 'dataset.yaml')
        indices = SimpleNamespace(i2w={}, w2i={})
        
        with open(yaml_abspath, 'w') as f_yaml:
            f_yaml.write(f'path: {output_folder}\n')
            f_yaml.write('train: images/train\n')
            f_yaml.write('val: images/validation\n')
            f_yaml.write('test: images/test\n')

            f_yaml.write('\n')
            f_yaml.write('#Classes\n')
            f_yaml.write('names:\n')
            for label in dictionary.label_list:
                index = dictionary.index_of(label)
                f_yaml.write(f'  {index}: {label}\n')
                indices.i2w[index] = label
                indices.w2i[label] = index
        return indices

    def transcode(self, output_folder: str):
        """
        Transcode the MuRET dataset to YOLO format.
            :param output_folder: Output folder where the dataset will be saved
        """
        # Obtains the YOLO data information
        indices = self.__generate_dataYAML(output_folder)

        # Exporting dictionaries to JSON
        muret_dicts_abspath = os.path.join(output_folder, 'muret_dicts')
        os.makedirs(muret_dicts_abspath, exist_ok=True)

        for target in ['i2w', 'w2i']:
            abspath = os.path.join(muret_dicts_abspath, f'{target}.json')
            with open(abspath, 'w') as f:
                json.dump(getattr(indices, target), f)

    # def __create_object_to_detect_from_symbol(self, symbol):
    #     class_index = self.training_package.agnostic_symbol_types.index_of(symbol.agnostic_symbol_type)
    #     object_to_detect = ObjectToDetect(symbol.bounding_box, class_index)
    #     return object_to_detect

    def __create_object_to_detect_from_region(self, region: Region):
        class_index = self.training_package.region_types.index_of(region.type)
        object_to_detect = Object(class_index, region.bounding_box)
        return object_to_detect

    def __generate_full_image(self, image: RemoteImage) -> ImageYOLO:
        try:
            image_array = image.load_image(self.image_transcoder, width=self.resize[0], height=self.resize[1])
            page_class_index = self.training_package.region_types.index_of('page')
            objects_to_detect = []
            if self.isRegionsTranscode:
                for page in image.representation.pages:
                    objects_to_detect.append(Object(page_class_index, page.bounding_box))
                    for region in page.regions:
                        if region.type != 'undefined' and region.bounding_box is not None:
                            object_to_detect = self.__create_object_to_detect_from_region(region)
                            objects_to_detect.append(object_to_detect)
                            
            elif self.object_kind == otdk.SYMBOLS_IN_IMAGES:
                for page in image.representation.pages:
                    for region in page.regions:
                        self.__fill_objects_to_detect_in_region(objects_to_detect, region)

            return ImageYOLO(image.file_name, image_array, objects_to_detect, image.dimensions())
        except Exception as e: # TODO: Catch specific exceptions
            print(e)
            logging.warning(f'Cannot retrieve image {image.url}')
            return None

    def _generate_full_images(self) -> list:
        logging.info('Generating full images')
        result = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: #TODO- Concurrence removed
            futures = []
            for original_image in self.original_images:
                future_staff_image = executor.submit(self.__generate_full_image, original_image)
                futures.append(future_staff_image)

            for future in concurrent.futures.as_completed(futures):
                full_image = future.result()
                if full_image:
                    result.append(full_image)
        return result

    def _generate_images_from_staves(self) -> list:
        logging.info('Generating images from staves')
        muret_image: ImageMuRET

        result = []  # list of ImageForYOLO
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for original_image in self.original_images:
                muret_image = original_image.muret_image
                remote_image = original_image.remote_image

                remote_image_dimensions = remote_image.dimensions()
                ratio_dimensions = self.__compute_ratio(remote_image_dimensions.width, remote_image_dimensions.height)

                for page in muret_image.pages:
                    for region in page.regions:
                        if region.region_type == 'staff' and len(region.symbols) > 0:
                            future_staff_image = executor.submit(self.__generate_region, remote_image, region, ratio_dimensions)
                            futures.append(future_staff_image)

            for future in concurrent.futures.as_completed(futures):
                staff_image = future.result()
                result.append(staff_image)
        return result

    def __save_image(self, image: ImageYOLO, images_folder: str, labels_folder: str):
        # Save the PNG image
        absolute_image_filename = os.path.join(images_folder, image.file_name)
        self.image_transcoder.save_as_png(absolute_image_filename, image.image_array)

        # Generate the labels
        with open(os.path.join(labels_folder, image.file_name + '.txt'), 'w') as f:
            f.write(str(image))
    
    def _save_files(self, images: List[ImageYOLO]):
        partitions = PartitionManager.partition(images, self.splits)
        
        # Iterate partitons using key as kind and value as list of images
        for kind, images in partitions.__dict__.items():
            images_folder = getattr(self, f'images.{kind}')
            labels_folder = getattr(self, f'labels.{kind}')

            # Concurrently save images and labels
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for image in images:
                    executor.submit(self.__save_image, image, images_folder, labels_folder)

