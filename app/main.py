from app.image_builder import ImageSettings, ImageBuilder
from app.path_utils import IMG_DIR


def build_test_image_settings_01():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.0)
    settings.pins = 100
    settings.min_threads = 400
    settings.max_threads = 800
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.5

    builder = ImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_02():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.0)
    settings.pins = 100
    settings.min_threads = 800
    settings.max_threads = 800
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.25

    builder = ImageBuilder(img_path, settings)
    builder.build()


def build_test_image_settings_03():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.2)
    settings.pins = 200
    settings.min_threads = 1000
    settings.max_threads = 1000
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.11

    builder = ImageBuilder(img_path, settings)
    builder.build()


if __name__ == "__main__":
    build_test_image_settings_03()
