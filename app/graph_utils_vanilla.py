from typing import List, Tuple


def draw_antialiased_line(
    m: List[List[float]],
    a: Tuple[int, int],
    b: Tuple[int, int],
    intensity: float = 1.0,
) -> None:
    """
    Draws an anti-aliased line from point a to point b on the matrix m.

    Parameters:
        m (List[List[float]]): The matrix to draw the line on.
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
        if 0 <= y < len(m) and 0 <= x < len(m[0]):
            m[y][x] += value

    for x in range(x1, x2 + 1):
        y_int = int(y)
        frac = y - y_int

        # Draw the fractional parts for anti-aliasing
        set_intensity(x, y_int, (1 - frac) * intensity)
        set_intensity(x, y_int + 1, frac * intensity)

        y += gradient


def print_matrix_1_channel(m: List[List[float]]) -> None:
    print()
    for row in m:
        for cell in row:
            print(f"{cell:.1f}", end=" ")
        print()