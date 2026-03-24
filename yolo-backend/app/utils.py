import cv2
import numpy as np
import base64
from PIL import Image
import io

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Convert uploaded file bytes to OpenCV image."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Make sure it is a valid JPG or PNG.")
    return img

def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string for sending to frontend."""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"

def get_confidence_color(confidence: float) -> str:
    """Return a color label based on confidence level."""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"