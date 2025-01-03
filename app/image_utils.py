from typing import List

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from PIL.Image import Resampling
from scipy.ndimage import sobel

from app.constants import ImageFilters


def preprocess_image(
        input_path: str,
        output_path: str,
        target_size: int = 0,
        brightness: float = 1.0,
        contrast: float = 1.0,
        filters: List[ImageFilters] = []) -> None:
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
        filters (ImageFilters): image filters to apply (default is []).
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

    if ImageFilters.Sobel in filters:
        # - apply Sobel filter to the image and invert it
        image = apply_sobel_filter(image)
        # invert
        image = Image.fromarray(255 - np.array(image))

    # Step 5: Mask out the area outside the central circle
    mask = Image.new("L", (square_size, square_size), color=0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, square_size, square_size), fill=255)

    image = Image.composite(image, Image.new("L", image.size, color=255), mask)

    # Step 6: Save the resulting image
    image.save(output_path)

# Example usage:
# preprocess_image("input.png", "output.png", brightness=1.2, contrast=0.8)

def apply_sobel_filter(image: Image) -> Image:
    image_array = np.array(image)

    # Ensure the image is grayscale (2D array)
    if image_array.ndim == 3:  # Convert RGB to grayscale if needed
        image_array = image_array.mean(axis=2)

    # Apply the Sobel filter along the x and y axes
    sobel_x = sobel(image_array, axis=1)  # Horizontal edges
    sobel_y = sobel(image_array, axis=0)  # Vertical edges

    # Combine the Sobel filtered results to get the edge magnitude
    sobel_magnitude = np.hypot(sobel_x, sobel_y)

    # Normalize the result to the range [0, 1] for visualization
    sobel_magnitude = (sobel_magnitude / np.max(sobel_magnitude)) if np.max(sobel_magnitude) > 0 else sobel_magnitude

    # Convert the result back to an image for visualization
    sobel_image = Image.fromarray((sobel_magnitude * 255).astype(np.uint8))
    return sobel_image

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
