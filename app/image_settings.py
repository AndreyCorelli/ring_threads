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
