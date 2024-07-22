from .AgnosticSymbol import AgnosticSymbol
from .Page import Page
from .Region import Region
from .Image import Image
from .Dictionary import Dictionary
from .Package import Package, PackageFilesLoader

from enum import Enum
ObjectsToDetectKind = Enum('ObjectKind', ['REGIONS', 'SYMBOLS_IN_REGIONS', 'SYMBOLS_IN_IMAGES'])