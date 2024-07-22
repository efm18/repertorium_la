class MuRETException(Exception):
    """Exception raised for errors on datasets.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        super().__init__(message)