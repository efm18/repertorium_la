import json
import logging

from exceptions import MuRETException

class Dictionary:
    """
    It contains a list of labels and an inverted dictionary to obtain the index of a label.
    """

    label_list: list[str]
    inverted: dict[str, int]

    def __init__(self, label_list: list[str] = []):
        """
        It creates an empty dictionary
        """
        self.label_list = []
        self.inverted = {}
        for label in label_list:
            self.add(label)

    @classmethod
    def from_json(cls: type, dictionary_json_file: str, root_element: str):
        """
        :param dictionary_json_file: JSON file containing the dictionary. The same file may contain several dictionaries
        :param root_element: Root element that contains the dictionary
        :returns Dictionary
        """
        logging.info(f'Reading dictionary from {dictionary_json_file} with root element {root_element}')
        with open(dictionary_json_file, 'r') as f:
            json_object = json.loads(f.read())
        if root_element not in json_object:
            raise MuRETException(f"No root element named {root_element}'")
        label_list = json_object[root_element]
        return cls(label_list)

    def add(self, label):
        """
        It adds the label to the dictionary if it does not exist

        :param label: Label to add
        """
        if label not in self.label_list:
            self.label_list.append(label)
            self.inverted[label] = self.label_list.index(label)

    def size(self) -> int:
        """
        :return: The number of labels in the dictionary
        """
        return len(self.label_list)

    def contains(self, label: str) -> bool:
        """
        It checks whether the label exists in the dictionary

        :param label: Label to search
        :return: True if it exists
        """
        return label in self.inverted

    def index_of(self, label: str) -> int:
        """
        It looks for the label in the "inverted" dictionary and obtains the index of the label

        :param label: Label to search
        """
        if label not in self.inverted:
            raise MuRETException(f'Label "{label}" not found in dictionary')
        return self.inverted[label]

    def label(self, index: int) -> str:
        """
        It retrieves a label given the integer index

        :param index:           Integer index
        :raises MuRETException: If the index is negative, greater than the number of labels 
                                or not found in the dictionary
        """
        if index < 0:
            raise MuRETException(f'Negative index {index}')
        if index >= len(self.label_list):
            raise MuRETException(f'Index >= {len(self.label_list)}')
        result = self.label_list[index]
        if result is None:
            raise MuRETException(f'Index not found in dictionary {index}')
        return result