from app.image_builder import ImageBuilder
from app.stochastic_image_builder import StochasticImageBuilder
from app.image_settings import ImageSettings
from app.path_utils import IMG_DIR


def build_test_image_settings_01():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.0)
    settings.pins = 100
    settings.min_threads = 400
    settings.max_threads = 800
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.5

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_02():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.0)
    settings.pins = 100
    settings.min_threads = 800
    settings.max_threads = 800
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.25

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_03():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.2)
    settings.pins = 200
    settings.min_threads = 1000
    settings.max_threads = 1000
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.11

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_04():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.2)
    settings.pins = 240
    settings.min_threads = 3054
    settings.max_threads = 3054
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.09

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_05():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=300, brightness=1.0, contrast=1.6)
    settings.pins = 240
    settings.min_threads = 3054
    settings.max_threads = 3054
    settings.threads_in_iteration = 1200
    settings.thread_intensity = 0.09

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_sorted_01():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.2)
    settings.pins = 240
    settings.min_threads = 3054
    settings.max_threads = 3054
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.09

    builder = ImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_06():
    img_path = IMG_DIR + "/guinea_2.jpeg"
    settings = ImageSettings(size=300, brightness=1.0, contrast=1.2)
    settings.pins = 240
    settings.min_threads = 2800
    settings.max_threads = 2800
    settings.threads_in_iteration = 1000
    settings.thread_intensity = 0.09

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_07():
    img_path = IMG_DIR + "/2.png"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.2)
    settings.pins = 240
    settings.min_threads = 3054
    settings.max_threads = 3054
    settings.threads_in_iteration = 1200
    settings.thread_intensity = 0.09

    builder = StochasticImageBuilder(img_path, settings)
    builder.build()


def build_test_sorted_02():
    img_path = IMG_DIR + "/2.png"
    settings = ImageSettings(size=700, brightness=1.0, contrast=1.2)
    settings.pins = 240
    settings.min_threads = 2100
    settings.max_threads = 2100
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.09

    builder = ImageBuilder(img_path, settings)
    builder.build()


if __name__ == "__main__":
    build_test_image_settings_05()
