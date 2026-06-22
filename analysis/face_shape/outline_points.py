import cv2
import numpy as np

from analysis.face_landmarks import get_face_landmark_points


FACE_OVAL_LANDMARKS = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No face contour was detected.")
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def get_face_oval_landmarks(image_bgr, landmark_points=None):
    if landmark_points is None:
        landmark_points = get_face_landmark_points(image_bgr)

    points = []
    for index in FACE_OVAL_LANDMARKS:
        x, y = landmark_points[index][:2]
        x = int(round(float(x)))
        y = int(round(float(y)))
        points.append((x, y))
    return np.array(points, dtype=np.int32)


def project_landmarks_to_contour(landmark_points, contour_points):
    contour = contour_points.astype(np.float32)
    projected = []

    for point in landmark_points.astype(np.float32):
        distances = np.sum((contour - point) ** 2, axis=1)
        nearest = contour_points[int(np.argmin(distances))]
        projected.append((int(nearest[0]), int(nearest[1])))

    return projected
