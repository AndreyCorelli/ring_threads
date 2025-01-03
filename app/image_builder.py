import os
from math import pi, sin, cos
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

from app.graph_utils import draw_antialiased_line, get_line_rel_intensity
from app.image_settings import ImageSettings
from app.image_utils import preprocess_image, image_to_normalized_array

Coord = tuple[int, int]


class ImageBuilder:
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
        self.diff = float("inf")  # the difference between the two images

        self.pins: List[Coord] = []
        self.pins_count = settings.pins
        self.pins_brightness: Dict[Tuple[int, int], float] = {}
        self._place_pins()

    def build(self):
        preprocess_image(self.orig_path, self.preproc_path,
                         self.settings.size, self.settings.brightness,
                         self.settings.contrast, self.settings.filters)
        self.target_matrix = image_to_normalized_array(self.preproc_path)
        self.target_matrix = 1.0 - self.target_matrix  # invert the image
        self.matrix: np.array = np.zeros((self.size, self.size))
        print(f"Image preprocessed & loaded: {self.preproc_path}")
        self._draw_lines()
        self._save_result()

    def _save_result(self) -> None:
        # invert it back
        self.matrix = 1.0 - self.matrix
        self.matrix = np.clip(self.matrix, 0.0, 1.0)
        self.matrix = (self.matrix * 255).astype(np.uint8)
        Image.fromarray(self.matrix).save(self.result_path)

    def _draw_lines(self) -> None:
        self._calc_pins_brightness()
        # sort the pins' pairs by brightness
        sorted_pins = sorted(self.pins_brightness, key=lambda x: self.pins_brightness[x], reverse=True)
        # draw lines between the pins
        for a, b in sorted_pins[:self.settings.max_threads]:
            self._draw_thread(a, b)

    def _calc_pins_brightness(self) -> None:
        # for each pair of pins calculate the relative brightness of the line between them
        processed, interval = 0, 200
        for a, b in self.pins_brightness:
            pin_a, pin_b = self.pins[a], self.pins[b]
            brightness = get_line_rel_intensity(self.target_matrix, pin_a, pin_b)
            self.pins_brightness[(a, b)] = brightness

            processed += 1
            if processed == interval:
                print(f"Processed {processed} / {len(self.pins_brightness)} pin pairs")
                processed = 0

    def _place_pins(self):
        # places self.pins_count pins on the circumference of a circle
        center = self.size // 2
        radius = self.size // 2

        for i in range(self.pins_count):
            angle = 2 * pi * i / self.pins_count
            x = center + radius * sin(angle)
            y = center - radius * cos(angle)
            self.pins.append((int(round(x)), int(round(y))))

        min_step = self.settings.min_pin_distance

        for i in range(self.pins_count):
            for j in range(i + min_step, self.pins_count + min_step):
                a, b = i, j
                if b >= self.pins_count:
                    b -= self.pins_count
                self.pins_brightness[tuple(sorted([a, b]))] = -1.0

    def _draw_thread(self, a: int, b: int) -> None:
        draw_antialiased_line(self.matrix, self.pins[a], self.pins[b], self.settings.thread_intensity)