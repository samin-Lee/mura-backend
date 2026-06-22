import math

import cv2
import numpy as np

from analysis.face_landmarks import get_face_landmark_points


def get_eye_centers(image_bgr):
    landmarks = get_face_landmark_points(image_bgr)

    # MediaPipe FaceMesh eye corner landmarks.
    left_eye = (landmarks[33][:2] + landmarks[133][:2]) / 2.0
    right_eye = (landmarks[362][:2] + landmarks[263][:2]) / 2.0
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
