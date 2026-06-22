from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from urllib.request import urlretrieve

import cv2
import numpy as np


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_FACE_LANDMARKER_MODEL_PATH = Path("data/models/face_landmarker.task")
_LANDMARKER_LOCK = Lock()
_MODEL_DOWNLOAD_LOCK = Lock()


def get_face_landmarker_model_path() -> str:
    model_path = Path(
        os.getenv(
            "MEDIAPIPE_FACE_LANDMARKER_MODEL_PATH",
            DEFAULT_FACE_LANDMARKER_MODEL_PATH,
        )
    )
    if model_path.exists():
        return str(model_path)

    with _MODEL_DOWNLOAD_LOCK:
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
    return str(model_path)


def load_mediapipe_tasks():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install it with: pip install mediapipe"
        ) from exc
    return mp


@lru_cache(maxsize=1)
def get_face_landmarker():
    mp = load_mediapipe_tasks()
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=get_face_landmarker_model_path()),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_face_landmarks(image_bgr):
    if image_bgr is None:
        raise ValueError("Image is empty")

    mp = load_mediapipe_tasks()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    with _LANDMARKER_LOCK:
        result = get_face_landmarker().detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No MediaPipe face landmarks were detected.")

    return result.face_landmarks[0]


def get_face_landmark_points(image_bgr):
    if image_bgr is None:
        raise ValueError("Image is empty")

    height, width = image_bgr.shape[:2]
    landmarks = detect_face_landmarks(image_bgr)
    return np.array(
        [
            (
                landmark.x * (width - 1),
                landmark.y * (height - 1),
                landmark.z * max(width, height),
            )
            for landmark in landmarks
        ],
        dtype=np.float32,
    )
