from .Page import Page

class Image:
    """
    It contains the representation of an image in a JSON file from MuRET
    Attributes:
        :ID: unique ID in MuRET
        :url: URL of the image
        :filename: original file name
        :pages: Division of the image into pages
    """

    ID: int
    url: str
    filename: str
    pages: list[Page]

    def __init__(self, unique_id: int, url: str, filename: str):
        self.ID = unique_id
        self.filename = filename
        self.url = url
        self.pages = [] # initialized as empty

    def add_page(self, page: Page):
        self.pages.append(page)