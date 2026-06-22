from functools import lru_cache

import cv2
import numpy as np

from analysis.face_shape.model_loader import get_face_parsing_model_path


FACE_LABELS = {
    1,   # skin
    2,   # left brow
    3,   # right brow
    4,   # left eye
    5,   # right eye
    6,   # glasses
    10,  # nose
    11,  # mouth
    12,  # upper lip
    13,  # lower lip
}


def _input_hw(input_shape):
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    height, width = input_shape[2], input_shape[3]
    if isinstance(height, str) or height is None:
        height = 512
    if isinstance(width, str) or width is None:
        width = 512
    return int(height), int(width)


def preprocess(image_bgr, size):
    input_h, input_w = size
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
    return tensor.astype(np.float32)


def parse_face(session, image_bgr):
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_size = _input_hw(input_meta.shape)

    tensor = preprocess(image_bgr, input_size)
    output = session.run(None, {input_name: tensor})[0]

    if output.ndim == 4:
        parsing = np.argmax(output[0], axis=0).astype(np.uint8)
    elif output.ndim == 3:
        parsing = np.argmax(output, axis=0).astype(np.uint8)
    elif output.ndim == 2:
        parsing = output.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported output shape: {output.shape}")

    image_h, image_w = image_bgr.shape[:2]
    return cv2.resize(parsing, (image_w, image_h), interpolation=cv2.INTER_NEAREST)


def make_face_mask(parsing):
    mask = np.isin(parsing, list(FACE_LABELS)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


@lru_cache(maxsize=2)
def _create_cached_session(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required. Install it with: pip install onnxruntime"
        ) from exc

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def create_session(model_path):
    model_path = model_path or get_face_parsing_model_path()
    return _create_cached_session(str(model_path))
