import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.face_shape.draw_face_outline import (
    DEFAULT_MODEL_PATH,
    create_session,
    draw_largest_outline,
    make_face_mask,
    parse_face,
)


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


def default_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}_outline_points{input_path.suffix}")


def load_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install it with: pip install mediapipe"
        ) from exc
    return mp


def get_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No face contour was detected.")
    return max(contours, key=cv2.contourArea).reshape(-1, 2)


def get_face_oval_landmarks(image_bgr):
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
    points = []
    for index in FACE_OVAL_LANDMARKS:
        landmark = landmarks[index]
        x = int(round(landmark.x * (width - 1)))
        y = int(round(landmark.y * (height - 1)))
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


def draw_points(image_bgr, points, color, radius, show_index):
    result = image_bgr.copy()
    for number, (x, y) in enumerate(points):
        cv2.circle(result, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            cv2.putText(
                result,
                str(number),
                (x + radius + 2, y - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return result


def save_points(points, path):
    data = [{"index": index, "x": x, "y": y} for index, (x, y) in enumerate(points)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Draw MediaPipe-guided points on the face parsing outline created from "
            "draw_face_outline.py logic."
        )
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Face parsing ONNX model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path. Defaults to '<input>_outline_points.<ext>'.",
    )
    parser.add_argument(
        "--points-json",
        type=Path,
        help="Optional JSON path for saving projected outline point coordinates.",
    )
    parser.add_argument("--radius", type=int, default=3, help="Point radius")
    parser.add_argument(
        "--point-color",
        nargs=3,
        type=int,
        default=(0, 0, 255),
        metavar=("B", "G", "R"),
        help="Point color in BGR order. Default: 0 0 255",
    )
    parser.add_argument(
        "--outline-color",
        nargs=3,
        type=int,
        default=(0, 255, 0),
        metavar=("B", "G", "R"),
        help="Outline color in BGR order. Default: 0 255 0",
    )
    parser.add_argument("--outline-thickness", type=int, default=1)
    parser.add_argument("--show-index", action="store_true", help="Draw point numbers")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    session = create_session(args.model)
    parsing = parse_face(session, image)
    mask = make_face_mask(parsing)
    contour = get_largest_contour(mask)
    landmarks = get_face_oval_landmarks(image)
    outline_points = project_landmarks_to_contour(landmarks, contour)

    result = draw_largest_outline(
        image,
        mask,
        tuple(args.outline_color),
        args.outline_thickness,
    )
    result = draw_points(
        result,
        outline_points,
        tuple(args.point_color),
        args.radius,
        args.show_index,
    )

    output_path = args.output or default_output_path(args.image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise SystemExit(f"Failed to write output image: {output_path}")

    if args.points_json:
        save_points(outline_points, args.points_json)

    print(f"Saved: {output_path}")
    print(f"Points: {len(outline_points)}")
    if args.points_json:
        print(f"Saved points: {args.points_json}")


if __name__ == "__main__":
    main()
