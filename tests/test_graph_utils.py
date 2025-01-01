import numpy as np

from app.graph_utils import draw_antialiased_line, print_matrix_1_channel, get_line_rel_intensity


def test_draw_line_1_0():
    # h*w = 3x7 matrix
    m = np.zeros((3, 7))
    draw_antialiased_line(m, (0, 2), (6, 0), 1.0)

    assert round(m[2, 0], 2) == 1.0
    assert round(m[0, 6], 2) == 1.0

    print_matrix_1_channel(m)


def test_draw_line_0_5():
    # h*w = 3x7 matrix
    m = np.zeros((3, 7))
    draw_antialiased_line(m, (0, 2), (6, 0), 0.5)

    assert round(m[2, 0], 2) == 0.5
    assert round(m[0, 6], 2) == 0.5

    print_matrix_1_channel(m)


def test_get_zero_intensity():
    m = np.zeros((3, 7))
    a, b = (0, 2), (6, 0)
    intensity = get_line_rel_intensity(m, a, b)
    assert round(intensity, 2) == 0.0


def test_get_intensity_0_6():
    m = np.zeros((3, 7))
    a, b = (0, 2), (6, 0)
    draw_antialiased_line(m, a, b, 0.6)

    intensity = get_line_rel_intensity(m, a, b)
    assert round(intensity, 2) == 0.6
