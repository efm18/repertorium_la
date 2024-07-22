import os
import json
import logging
import concurrent.futures

from exceptions import MuRETException
from .Dictionary import Dictionary
from .Page import Page
from .Region import Region
from .Image import Image
from .AgnosticSymbol import AgnosticSymbol
from ..YOLO.BoundigBox import BoundingBox

from operator import attrgetter

class DatasetFile:
    """
    It contains a file with the decoded contents of the corresponding JSON file.
    """

    folder_structure: str
    name: str
    json_contents: dict

    def __init__(self, folder_structure, name_without_extension, json_contents):
        self.folder_structure = folder_structure
        self.name = name_without_extension
        self.json_contents = json_contents

class PackageFilesLoader:
    """
    It contains the representation of the file structure of the MuRET downloaded training package
    """

    def __init__(self, file_or_folder: str):
        """
        :param file_or_folder:  It contains a JSON file or a folder containing, maybe recursively, a set of JSON files.
        :raises IOError:        When the input file does not exist.
        :raises MuRETException: When the input folder is empty.
        """
        if not os.path.exists(file_or_folder):
            raise IOError("File or folder '" + file_or_folder + "' does not exist")
        self.__load(file_or_folder)
        self.files.sort(key=attrgetter('folder_structure', 'name'))

    @classmethod
    def load(cls: type, inputPath: str) -> list[DatasetFile]:
        """
        It reads a set of folders and subfolders (or a single file) containing JSON files and 
        returns a list of DatasetFile objects. The list is sorted by folder_structure and name.

        :param input:       Either a JSON file or a folder containing JSON files
        :raises IOError:    When the input folder does not exist
        """
        if not os.path.exists(inputPath):
            raise IOError("File or folder '" + inputPath + "' does not exist")
        
        logging.info(f'Loading MuRET training package from {inputPath}')

        if os.path.isdir(inputPath): # if it is a folder
            files = cls._load_folder(inputPath)
        else: # if it is a file
            files = [cls._load_json(None, inputPath)]

        logging.debug(f'{len(files)} JSON files read')
        files.sort(key=attrgetter('folder_structure', 'name'))
        return files
    
    @classmethod
    def _load_folder(cls: type, folderPath: str) -> list[DatasetFile]:
        """
        It reads a set of folders and subfolders containing JSON files and returns a list of DatasetFile objects.
        The list is sorted by folder_structure and name.
        :param folder: Folder containing JSON files
        """
        files = []
        prefix_length = len(folderPath) + 1  # +1 for the '/' at the beginning
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for root, _, all_files in os.walk(folderPath):
                for filename in all_files:
                    fname = os.path.join(root, filename)
                    if fname.endswith('.json'):
                        future_file = executor.submit(cls._load_json, root[prefix_length:], fname)
                        futures.append(future_file)
            for future in concurrent.futures.as_completed(futures):
                training_set_file = future.result()
                files.append(training_set_file)
            executor.shutdown()
        return files

    @classmethod
    def _load_json(cls: type, folder_structure: str | None, json_file: str) -> DatasetFile:
        """
        It reads and decodes the file if it is a .JSON, or nothing elsewere, storing the decoded JSON into self.files
        :param folder_structure: The folder structure
        :param json_file:        The JSON path to read from.
¡       """
        with open(json_file, 'r') as f:
            contents = json.loads(f.read())
        absolute_path_without_extension = os.path.splitext(json_file)[0]
        _, tail = os.path.split(absolute_path_without_extension)
        name_without_extension = tail
        training_set_file = DatasetFile(folder_structure, name_without_extension, contents)
        return training_set_file

class Package:
    """
    It reads the JSON package downloaded from MuRET, uncompressed to a folder and converts it into
    a MuRETTrainingImages object.
    """

    region_types: Dictionary
    agnostic_symbol_types: Dictionary
    agnostic_positions_in_staff: Dictionary
    training_set_images: list[Image]

    def __init__(self, folder: str):
        """
        It reads the JSON package downloaded from MuRET, uncompressed to a folder and converts it into
        a MuRETTrainingImages object.

        :param folder: Folder where the JSONs are located
        :raises IOError: when the input folder does not exist
        """
        logging.info(f'Reading MuRET training jsom package from folder {folder}')

        if not os.path.exists(folder): 
            raise IOError(f'Folder {folder} does not exist')
        
        absolute_dictionary_file_name = os.path.join(folder, 'dictionary.json')
        if not os.path.exists(absolute_dictionary_file_name):
            raise IOError(f'Dictionary file {absolute_dictionary_file_name} does not exist')

        self.region_types = Dictionary.from_json(absolute_dictionary_file_name, 'region_dictionary')
        self.agnostic_symbol_types = Dictionary.from_json(absolute_dictionary_file_name, 'agnostic_symbol_types')
        self.agnostic_positions_in_staff = Dictionary.from_json(absolute_dictionary_file_name, 'agnostic_positions_in_staff')

        absolute_files = os.path.join(folder, 'files')
        muret_trainingset_package_files = PackageFilesLoader.load(absolute_files)
        if len(muret_trainingset_package_files) == 0:
            raise MuRETException('Empty MuRET training set package')

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            self.training_set_images = []
            for file in muret_trainingset_package_files:
                future = executor.submit(self.read_file, file,
                                         self.agnostic_symbol_types,
                                         self.agnostic_positions_in_staff)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                muret_image = future.result()
                self.training_set_images.append(muret_image)
            executor.shutdown()
            
        logging.info(f'#{(len(self.training_set_images))} read from input folder {folder}')

    def read_file(self, file: str, agnostic_symbol_types: Dictionary,
                  agnostic_positions_in_staff: Dictionary) -> Image:
        """
        It reads the contents of the JSON file. The method checks the contents are valid according to the dictionary.
        :param agnostic_symbol_types: Dictionary of agnostic symbol types.
        :param agnostic_positions_in_staff: Dictionary of positions in staff
        :param file: Input JSON file containing a json_contents property
        :return: An image object
        """
        unique_id = file.json_contents['id']
        url = file.json_contents['url']
        filename = file.json_contents['filename']
        muret_image = Image(unique_id, url, filename)
        if 'pages' in file.json_contents:
            for page in file.json_contents['pages']:
                # From page['bounding_box'] dictionary obtain from_x, from_y, to_x, to_y
                page_bounding_box = BoundingBox.from_MuRET(page['bounding_box'])
                muret_page = Page(page_bounding_box)
                muret_image.add_page(muret_page)
                if 'regions' in page:
                    for region in page['regions']:
                        region_bounding_box = BoundingBox.from_MuRET(region['bounding_box'])
                        region_type = region['type']
                        region_semantic_encoding = None
                        if 'semantic_encoding' in region:
                            region_semantic_encoding = region['semantic_encoding']

                        if not self.region_types.contains(region_type):
                            self.region_types.add(region_type)
                            raise MuRETException(f'Region type "{region_type}" not found in dictionary. A new entry will be created.')

                        muret_region = Region(region_type, region_bounding_box, region_semantic_encoding)
                        muret_page.add_region(muret_region)
                        if 'symbols' in region:
                            symbol_regions = region['symbols']
                            logging.info(f'Processing {len(symbol_regions)} symbols in region')
                            for symbol in region['symbols']:
                                muret_symbol_bounding_box = None
                                if 'bounding_box' in symbol:
                                    muret_symbol_bounding_box = BoundingBox.from_MuRET(symbol['bounding_box'])

                                muret_symbol_approximate_x = None
                                if 'approximate_x' in symbol:
                                    muret_symbol_approximate_x = symbol[muret_symbol_approximate_x]

                                muret_agnostic_symbol_type = symbol['agnostic_symbol_type']
                                if not agnostic_symbol_types.contains(muret_agnostic_symbol_type):
                                    self.agnostic_symbol_types.add(muret_agnostic_symbol_type)
                                    raise MuRETException(f'Agnostic symbol type "{muret_agnostic_symbol_type}" '
                                                            f'not found in dictionary. A new entry will be created.')
                                
                                muret_position_in_staff = symbol['position_in_staff']
                                if not agnostic_positions_in_staff.contains(muret_position_in_staff):
                                    self.agnostic_positions_in_staff.add(muret_position_in_staff)
                                    raise MuRETException(f'Position in staff "{muret_position_in_staff}" '
                                                            f'not found in dictionary. A new entry will be created.')

                                muret_symbol = AgnosticSymbol(muret_agnostic_symbol_type,
                                                                    muret_position_in_staff,
                                                                    muret_symbol_bounding_box,
                                                                    muret_symbol_approximate_x)
                                muret_region.add_symbol(muret_symbol)
        return muret_image
    
    def save():
        pass