import cv2
import numpy as np


def decode_image(image_bytes: bytes):
    image_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("?대?吏 ?붿퐫???ㅽ뙣")
    return image
