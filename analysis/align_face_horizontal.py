import math

import cv2
import numpy as np

from analysis.face_landmarks import load_mediapipe

def get_eye_centers(image_bgr):
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

    def point(index):
        landmark = landmarks[index]
        return np.array(
            [
                landmark.x * (width - 1),
                landmark.y * (height - 1),
            ],
            dtype=np.float32,
        )

    # MediaPipe FaceMesh eye corner landmarks.
    left_eye = (point(33) + point(133)) / 2.0
    right_eye = (point(362) + point(263)) / 2.0
    return left_eye, right_eye


def rotate_image_keep_size(image_bgr, angle_degrees, center):
    height, width = image_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D(tuple(center), angle_degrees, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def align_face_horizontal(image_bgr):
    left_eye, right_eye = get_eye_centers(image_bgr)
    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = math.degrees(math.atan2(dy, dx))
    center = (left_eye + right_eye) / 2.0

    # If the right eye is lower than the left eye, angle is positive.
    # Rotating by the same positive angle levels the eye line in image coordinates.
    aligned = rotate_image_keep_size(image_bgr, angle, center)
    return aligned, angle
