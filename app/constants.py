from enum import Enum


Coord = tuple[int, int]


class ImageFilters(Enum):
    Sobel = "Sobel"