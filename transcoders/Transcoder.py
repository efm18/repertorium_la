import logging

from utils.image_tools import ImageTranscoder
from entities.MuRET import ObjectsToDetectKind as otdk

class EncodingTranscoder:
    def __init__(self, object_kind: str = otdk.REGIONS):
        """
            :param input_muret_training_set_folder: Training set where dictionary.json and files with JSONs is located
            :param cache: Temporary cache
        """
        if object_kind not in otdk:
            raise ValueError(f'Unsupported object kind: {object_kind}')
        self.isStaffWise = object_kind == otdk.SYMBOLS_IN_REGIONS
        self.isRegionsTranscode = object_kind == otdk.REGIONS

    def __call__(self, output_folder: str, transcode_opts: dict = {}):
        self._create_output_structure(output_folder)
        self.transcode(output_folder, **transcode_opts)
        self.__export_images(output_folder)
    
    def _create_output_structure(self, output_folder: str):
        """
        This function creates the output folder structure.
            :param output_folder: Output folder where the dataset will be saved
        """
        raise NotImplementedError

    def transcode(self, output_folder: str, **kwargs):
        """
        This function transcodes images to a specific output format.
            :param output_folder: Output folder where the dataset will be saved
        """
        raise NotImplementedError

    def __export_images(self, output_folder: str):
        """
            This function exports images to a specific folder structure, different from original format.
        """
        logging.info('Generating images...')
        images = self.__generate_images()

        logging.info('Saving files...')
        self._save_files(images)

    def __generate_images(self):
        """
        It generates the images for the YOLO dataset
        """
        return self._generate_images_from_staves() if self.isStaffWise else self._generate_full_images()
    
    def _generate_images_from_staves(self):
        raise NotImplementedError
    
    def _generate_full_images(self) -> list:
        raise NotImplementedError
    
    def _save_files(self, images: list):
        raise NotImplementedError