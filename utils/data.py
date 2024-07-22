import math
import random
import logging
import threading
import concurrent.futures
from types import SimpleNamespace
from typing import Generic, List, TypeVar

from exceptions.MuRETException import MuRETException
from utils.image import AbstractImage, RemoteImage

class PartitionManager:
    @staticmethod
    def partition(dataset, splits: SimpleNamespace) -> SimpleNamespace:
        """
        It performs a partition of the dataset into train, test, and validation
            :param dataset: Dataset to partition
            :param splits: Splits to partition the dataset
            :return: Tuple with the indexes of the dataset
        """
        logging.info(f'Partitioning training {splits.train*100}%, validation {splits.validation*100}%, test {splits.test*100}%...')
        sum_split = round(splits.train + splits.validation + splits.test, 2)
        if sum_split != 1:
            raise MuRETException(f"The sum of all splits should be 1, and it is {sum_split}")
        
        idx_split_1 = math.floor(len(dataset) * splits.train)
        idx_split_2 = idx_split_1 + math.floor(len(dataset) * splits.validation)

        train_list = dataset[:idx_split_1]
        validation_list = dataset[idx_split_1:idx_split_2]
        test_list = dataset[idx_split_2:]
        return SimpleNamespace(train=train_list, validation=validation_list, test=test_list)

T = TypeVar('T')

class ImageLoader(Generic[T]):
    def __new__ (cls, images: List[T], folder: str, remote: bool = True) -> List[AbstractImage[T]]:
        logging.info(f'Downloading images from URLs into cache folder {folder}')
        lock = threading.Lock()
        images = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []

            for image in images:
                future_original_image = executor.submit(cls.__download_image, lock, image, folder)
                futures.append(future_original_image)

            for future in concurrent.futures.as_completed(futures):
                try:
                    image = future.result()
                    images.append(image)
                except Exception as e:
                    print(f'Cannot retrieve image {e}')

        return images

    def __download_image(lock: threading.Lock, image: T, cache: str) -> AbstractImage[T]:
        lock.acquire()  # to avoid checking for the same file
        try:
            remote_image = RemoteImage[T](image, cache)
        finally:
            lock.release()
        return remote_image
