import os
from concurrent.futures import ThreadPoolExecutor
from math import pi, sin, cos
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

from app.constants import Coord
from app.graph_utils import draw_antialiased_line, get_connecting_points_coords
from app.image_settings import ImageSettings
from app.image_utils import preprocess_image, image_to_normalized_array

# NB: my laptop has 8 logical cores (sysctl -n hw.logicalcpu)
THREAD_COUNT = 6


class PinPair:
    def __init__(self, a: int, b: int, p_a: Coord, p_b: Coord):
        self.a = a
        self.b = b
        self.p_a = p_a
        self.p_b = p_b
        # ([rows], [columns])
        self.trail: Tuple[List[int], List[int]] = ([], [])
        self._build_trail()

    def _build_trail(self) -> None:
        line = get_connecting_points_coords(self.p_a, self.p_b)
        for c, r in line:
            self.trail[0].append(r)
            self.trail[1].append(c)


class StringDiffImageBuilder:
    def __init__(self, orig_path: str, settings: ImageSettings):
        self.orig_path = orig_path
        self.preproc_path = os.path.splitext(orig_path)[0] + "_pre.png"
        self.result_path = os.path.splitext(orig_path)[0] + "_res.png"
        self.settings = settings
        self.size = settings.size

        # this is the image we're approximating
        self.target_matrix: np.array = np.zeros((1, 1))
        # this is the current image we're working on
        self.matrix: np.array = np.zeros((1, 1))
        self.diff = float("-inf")  # the difference between the two images

        self.pins: List[Coord] = []
        self.pins_count = settings.pins
        self.shuffled_pin_pairs: List[PinPair] = []
        # target image pixels' brightness for each trail between each pin pair
        self._target_trail_brightness: Dict[Tuple[int, int], np.ndarray] = {}
        self._connected_pairs: List[Tuple[int, int]] = []
        self._place_pins()

    def build(self):
        preprocess_image(self.orig_path, self.preproc_path,
                         self.settings.size, self.settings.brightness,
                         self.settings.contrast, self.settings.filters)
        self.target_matrix = image_to_normalized_array(self.preproc_path)
        self.target_matrix = 1.0 - self.target_matrix  # invert the image

        for pair in self.shuffled_pin_pairs:
            t_v = self.target_matrix[pair.trail]
            self._target_trail_brightness[pair.a, pair.b] = t_v

        self.matrix: np.array = np.zeros((self.size, self.size))
        print(f"Image preprocessed & loaded: {self.preproc_path}")
        self._draw_lines()
        self._connect_all_pairs()
        self._save_result()

    def _connect_all_pairs(self) -> None:
        self.matrix: np.array = np.zeros((self.size, self.size))
        for a, b in self._connected_pairs:
            draw_antialiased_line(self.matrix, self.pins[a], self.pins[b], self.settings.thread_intensity)

    def _save_result(self) -> None:
        # invert it back
        self.matrix = 1.0 - self.matrix
        self.matrix = np.clip(self.matrix, 0.0, 1.0)
        self.matrix = (self.matrix * 255).astype(np.uint8)
        Image.fromarray(self.matrix).save(self.result_path)

    def _draw_lines(self) -> None:
        # draw lines until max_threads or max_threads' reached but the quality isn't getting better
        threads = 0
        while threads < self.settings.max_threads:
            print(f"Iteration {threads + 1} / [{self.settings.min_threads} .. {self.settings.max_threads}]")
            #if not self._pull_thread():
            #    break
            new_diff = self._pull_thread()
            if new_diff <= self.diff and threads >= self.settings.min_threads:
                print(f"Quality not improving ({self.diff} -> {new_diff}), stopping at {threads} threads")
                break
            self.diff = new_diff
            threads += 1

    def _pull_thread(self) -> float:
        # go through each pair of pins and calculate the difference
        # between the current matrix brightness and the target matrix brightness
        # along the trail, connecting the pins
        max_diff = float("-inf")
        connected_pair, path_to_draw = None, None

        np.random.shuffle(self.shuffled_pin_pairs)
        pin_pairs = self.shuffled_pin_pairs[:self.settings.threads_in_iteration]

        for pair in pin_pairs:
            t_v = self._target_trail_brightness[pair.a, pair.b]
            m_v = self.matrix[pair.trail]
            m_diff = np.sum(t_v - m_v)
            if m_diff > max_diff:
                max_diff = m_diff
                path_to_draw = pair.trail
                connected_pair = (pair.a, pair.b)

        self.matrix[path_to_draw] = 1.0
        self._connected_pairs.append(connected_pair)

        return max_diff

    def _place_pins(self):
        # places self.pins_count pins on the circumference of a circle
        center = self.size // 2
        radius = self.size // 2 - 1

        for i in range(self.pins_count):
            angle = 2 * pi * i / self.pins_count
            x = center + radius * sin(angle)
            y = center - radius * cos(angle)
            self.pins.append((int(round(x)), int(round(y))))

        min_step = self.settings.min_pin_distance

        pin_pairs: Dict[Tuple[int, int], PinPair] = {}

        for i in range(self.pins_count):
            for j in range(i + min_step, self.pins_count + min_step):
                a, b = i, j
                if b >= self.pins_count:
                    b -= self.pins_count
                pin_indx = tuple(sorted([a, b]))
                if pin_indx not in pin_pairs:
                    pin_pairs[pin_indx] = PinPair(a, b, self.pins[a], self.pins[b])
        self.shuffled_pin_pairs = list(pin_pairs.values())
