from PIL import Image

from app.image_utils import preprocess_image, image_to_normalized_array
from app.path_utils import IMG_DIR


def test_preprocess_image():
    input_img, output_img = IMG_DIR + "/guinea_orig.jpg", IMG_DIR + "/guinea_prep.png"
    preprocess_image(input_img, output_img, target_size=400, brightness = 1.2, contrast = 0.8)

    # load preprocessed image
    m = image_to_normalized_array(output_img)
    assert m.shape == (400, 400)
    assert m[0, 0] == 1.0  # top-left pixel is white
