import os
from math import pi, sin, cos
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from app.graph_utils import draw_antialiased_line
from app.image_utils import preprocess_image, image_to_normalized_array

Coord = tuple[int, int]


class ImageSettings:
    def __init__(
            self,
            size: int,
            brightness: float = 1.0,
            contrast: float = 1.0,
            pins: int = 100,
            min_threads: int = 0,
            max_threads: int = 0,
            thread_intensity: float = 0.6,
            threads_in_iteration: int = 0,
            min_pin_distance: int = 5,):
        self.size = size
        self.brightness = brightness
        self.contrast = contrast
        self.pins = pins

        if not min_threads:
            min_threads = int(round(pins**1.33))
        if not max_threads:
            max_threads = int(round(min_threads*1.35))

        self.min_threads = min_threads
        self.max_threads = max_threads
        self.thread_intensity = thread_intensity
        self.threads_in_iteration = threads_in_iteration
        self.min_pin_distance = min_pin_distance

    def __str__(self):
        return (f"size={self.size}, brightness={self.brightness}, contrast={self.contrast}, pins={self.pins}, " +
                f"min_threads={self.min_threads}, max_threads={self.max_threads}")

    def __repr__(self):
        return self.__str__()


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
        self.shuffled_pin_pairs: List[Tuple[int, int]] = []
        self._place_pins()

    def build(self):
        preprocess_image(self.orig_path, self.preproc_path,
                         self.settings.size, self.settings.brightness, self.settings.contrast)
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
        # draw lines until max_threads or max_threads' reached but the quality isn't getting better
        threads = 0
        while threads < self.settings.max_threads:
            print(f"Iteration {threads + 1} / [{self.settings.min_threads} .. {self.settings.max_threads}]")
            if not self._pull_thread():
                break
            new_diff, thread_pair = self._pull_thread()
            if new_diff >= self.diff and threads >= self.settings.min_threads:
                print(f"Quality not improving ({self.diff} -> {new_diff}), stopping at {threads} threads")
                break
            # draw the thread
            self._draw_thread(*thread_pair, self.matrix)
            self.diff = new_diff
            threads += 1

    def _pull_thread(self) -> Tuple[float, Optional[Tuple[int, int]]]:
        # return True if a thread was added
        # shuffle pins again
        np.random.shuffle(self.shuffled_pin_pairs)
        best_diff = float("inf")
        thread_pair = None

        tries = self.settings.threads_in_iteration if self.settings.threads_in_iteration else len(self.shuffled_pin_pairs)
        tries = min(tries, len(self.shuffled_pin_pairs))

        for i in range(tries):
            a, b = self.shuffled_pin_pairs[i]
            cur_diff = self._get_diff_after_thread(a, b)
            if cur_diff < best_diff:
                best_diff = cur_diff
                thread_pair = a, b
        return best_diff, thread_pair

    def _get_diff_after_thread(self, a: int, b: int) -> float:
        # return the difference between the target and the current image after adding a thread
        # copy the matrix
        m = self.matrix.copy()
        # draw the thread
        self._draw_thread(a, b, m)
        # calc diff between m and self.target_matrix
        diff = np.sum(np.abs(m - self.target_matrix))
        return diff

    def _draw_thread(self, a: int, b: int, m: np.array) -> None:
        draw_antialiased_line(m, self.pins[a], self.pins[b], self.settings.thread_intensity)

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
                self.shuffled_pin_pairs.append(tuple(sorted([a, b])))
        self.shuffled_pin_pairs = list(set(self.shuffled_pin_pairs))
        # np.random.shuffle(self.shuffled_pin_pairs)