import argparse
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install it with: pip install mediapipe"
        ) from exc
    return mp


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def iter_images(path):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return

    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_aligned{input_path.suffix}")


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


def output_for_folder(input_root, output_root, image_path):
    relative = image_path.relative_to(input_root)
    return output_root / relative.with_name(
        f"{relative.stem}_aligned{relative.suffix}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Rotate face images so both eyes are horizontally aligned."
    )
    parser.add_argument("input", type=Path, help="Input image or folder")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path for a file, or output folder for a folder.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    if args.input.is_file():
        output_path = args.output or default_output_path(args.input)
        image_paths = [args.input]
    else:
        output_root = args.output or Path("facetest/aligned")
        image_paths = list(iter_images(args.input))

    if not image_paths:
        raise SystemExit(f"No images found: {args.input}")

    for image_path in image_paths:
        image = read_image(image_path)
        if image is None:
            print(f"Skipped unreadable image: {image_path}")
            continue

        try:
            aligned, angle = align_face_horizontal(image)
        except Exception as exc:
            print(f"Failed: {image_path} ({exc})")
            continue

        if args.input.is_file():
            save_path = output_path
        else:
            save_path = output_for_folder(args.input, output_root, image_path)

        if not write_image(save_path, aligned):
            print(f"Failed to write output image: {save_path}")
            continue

        print(f"Saved: {save_path} (angle={angle:.2f})")


if __name__ == "__main__":
    main()
