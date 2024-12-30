from app.image_builder import ImageSettings, ImageBuilder
from app.path_utils import IMG_DIR


def test_build_image():
    img_path = IMG_DIR + "/guinea_orig.jpg"
    settings = ImageSettings(size=400, brightness=1.0, contrast=1.0)
    settings.pins =100
    settings.min_threads = 400
    settings.max_threads = 800
    settings.threads_in_iteration = 200
    settings.thread_intensity = 0.5

    builder = ImageBuilder(img_path, settings)
    builder.build()
