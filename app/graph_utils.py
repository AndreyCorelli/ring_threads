import numpy as np
from typing import Tuple, List

from app.constants import Coord


def draw_antialiased_line(
    m: np.ndarray,
    a: Tuple[int, int],
    b: Tuple[int, int],
    intensity: float = 1.0
) -> None:
    """
    Draws an anti-aliased line from point a to point b on the matrix m.

    Parameters:
        m (np.ndarray): The matrix to draw the line on (2D array of floats).
        a (Tuple[int, int]): Starting point of the line (x1, y1).
        b (Tuple[int, int]): Ending point of the line (x2, y2).
        intensity (float): Maximum intensity for cells fully crossed by the line.
    """
    x1, y1 = a
    x2, y2 = b

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        # Transpose line if it is steep
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    if x1 > x2:
        # Ensure left-to-right drawing
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    gradient = (y2 - y1) / (x2 - x1) if dx != 0 else 0
    y = y1

    def set_intensity(x, y, value):
        """Safely set the intensity value for a cell in the matrix."""
        if 0 <= y < m.shape[0] and 0 <= x < m.shape[1]:
            m[y, x] += value

    for x in range(x1, x2 + 1):
        y_int = int(y)
        frac = y - y_int

        # Draw the fractional parts for anti-aliasing
        set_intensity(x, y_int, (1 - frac) * intensity)
        set_intensity(x, y_int + 1, frac * intensity)

        y += gradient


def get_line_rel_intensity(
    m: np.ndarray,
    a: Tuple[int, int],
    b: Tuple[int, int],
) -> float:
    """
    Sums all (weighted) intensities of the cells crossed by the line from point a to point b.
    Divides the sum by the number of cells crossed by the line.

    Parameters:
        m (np.ndarray): The matrix to draw the line on (2D array of floats).
        a (Tuple[int, int]): Starting point of the line (x1, y1).
        b (Tuple[int, int]): Ending point of the line (x2, y2).
    """
    x1, y1 = a
    x2, y2 = b

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        # Transpose line if it is steep
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    if x1 > x2:
        # Ensure left-to-right drawing
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    gradient = (y2 - y1) / (x2 - x1) if dx != 0 else 0
    y = y1

    cells_count, intensity = 0.0, 0.0

    for x in range(x1, x2 + 1):
        y_int = int(y)
        frac = y - y_int
        frac_rem = 1 - frac

        if frac_rem:
            if 0 <= y_int < m.shape[0] and 0 <= x < m.shape[1]:
                intensity += m[y_int, x] * frac_rem
                cells_count += frac_rem

        if frac:
            if 0 <= y_int + 1 < m.shape[0] and 0 <= x < m.shape[1]:
                intensity += m[y_int + 1, x] * frac
                cells_count += frac

        y += gradient

    if not cells_count:
        return 0.0

    return intensity / cells_count


def print_matrix_1_channel(m: np.ndarray) -> None:
    print()
    for row in m:
        for cell in row:
            print(f"{cell:.1f}", end=" ")
        print()


def get_connecting_points_coords(a: Coord, b: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Calculate all points connecting two coordinates (a and b) using the Bresenham algorithm.

    Args:
        a (Tuple[int, int]): Starting point (x1, y1).
        b (Tuple[int, int]): Ending point (x2, y2).

    Returns:
        List[Tuple[int, int]]: List of all points on the line from a to b.
    """
    x1, y1 = a
    x2, y2 = b

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    while True:
        points.append((x1, y1))
        if (x1, y1) == (x2, y2):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points