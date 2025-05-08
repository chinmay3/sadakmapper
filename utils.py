import numpy as np
import cv2
from PIL import Image

IMAGE_SIZE = 256

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))  # Match training input size
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    if image_array.ndim == 2:  # grayscale
        image_array = np.stack([image_array]*3, axis=-1)
    image_array = image_array.astype(np.float32)
    return np.expand_dims(image_array, axis=0)  # Add batch dimension


def postprocess_mask(mask: np.ndarray) -> Image.Image:
    mask = np.squeeze(mask)  # Remove batch/channel dims
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold
    return Image.fromarray(mask)

