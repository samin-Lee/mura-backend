import cv2
import numpy as np


def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install it with: pip install mediapipe"
        ) from exc
    return mp


def get_face_landmark_points(image_bgr):
    if image_bgr is None:
        raise ValueError("Image is empty")

    mp = load_mediapipe()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        raise ValueError("No MediaPipe face landmarks were detected.")

    landmarks = result.multi_face_landmarks[0].landmark
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
