import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from PIL.Image import Resampling


def preprocess_image(
        input_path: str,
        output_path: str,
        target_size: int = 0,
        brightness: float = 1.0,
        contrast: float = 1.0) -> None:
    """
    Processes an image by performing the following steps:
    1. Loads an image from a file.
    2. Makes the image square by cutting the right or bottom part.
    3. Converts the image to grayscale.
    4. Adjusts brightness and contrast.
    5. Masks out the area outside a central circle with white.
    6. Saves the resulting image to another file.

    Parameters:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        target_size: Size of the output image (default is 0).
        brightness (float): Brightness adjustment factor (default is 1.0).
        contrast (float): Contrast adjustment factor (default is 1.0).
    """
    # Step 1: Load the image
    image = Image.open(input_path)

    # Step 2: Make the image square by cropping
    width, height = image.size
    square_size = min(width, height)
    image = image.crop((0, 0, square_size, square_size))

    # Step 2.5: Resize the image to the target size
    target_size = target_size or square_size
    square_size = target_size
    image = image.resize((target_size, target_size), Resampling.BICUBIC)

    # Step 3: Convert the image to grayscale
    image = image.convert("L")

    # Step 4: Adjust brightness and contrast
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)

    # Step 5: Mask out the area outside the central circle
    mask = Image.new("L", (square_size, square_size), color=0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, square_size, square_size), fill=255)

    image = Image.composite(image, Image.new("L", image.size, color=255), mask)

    # Step 6: Save the resulting image
    image.save(output_path)

# Example usage:
# preprocess_image("input.png", "output.png", brightness=1.2, contrast=0.8)


def image_to_normalized_array(image_path: str) -> np.ndarray:
    """
    Loads a grayscale image and converts it to a normalized numpy array.

    Parameters:
        image_path (str): Path to the grayscale image.

    Returns:
        np.ndarray: A 2D numpy array with values normalized to [0.0, 1.0].
    """
    # Load the image as grayscale (if not already)
    image = Image.open(image_path).convert("L")

    # Convert the image to a numpy array
    array = np.asarray(image, dtype=np.float32)

    # Normalize the values to the range [0.0, 1.0]
    normalized_array = array / 255.0

    return normalized_array

# Example usage
# result = image_to_normalized_array("grayscale_image.png")
