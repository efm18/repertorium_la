import os
import threading

lock = threading.Lock()


class SynchronizedDiskAccess:
    """
    It synchronizes operations of disk access
    """

    @staticmethod
    def create_if_not_exists(path: str):
        with lock:
            if not os.path.exists(path):
                os.makedirs(path)