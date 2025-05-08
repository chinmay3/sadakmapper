import numpy as np
import cv2
from PIL import Image

IMAGE_SIZE = 256

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image)
    image_array = image_array / 255.0  # normalize
    return np.expand_dims(image_array, axis=0)  # add batch dim

def postprocess_mask(mask: np.ndarray) -> Image.Image:
    mask = (mask > 0.5).astype(np.uint8) * 255
    return Image.fromarray(mask.squeeze(), mode="L")
