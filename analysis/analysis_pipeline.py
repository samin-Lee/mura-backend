from analysis.face_shape.side_jaw_classifier import classify_image as classify_face_shape
from analysis.image_decoder import decode_image
from analysis.personal_color.image_analyzer import extract_skin_region
from analysis.personal_color.lab_analyzer import calculate_lab_average
from analysis.personal_color.personal_color_analyzer import (
    classify_brightness,
    classify_season,
    classify_skin_tone,
)
from analysis.r2_image_loader import download_image_from_r2


async def analyze_image_from_r2(file_key: str) -> dict:
    image_bytes = await download_image_from_r2(file_key)
    image = decode_image(image_bytes)
    skin_region = extract_skin_region(image)
    lab_values = calculate_lab_average(skin_region)
    face_shape = classify_face_shape(image)

    return {
        "lab_values": lab_values,
        "tone": classify_skin_tone(lab_values),
        "brightness": classify_brightness(lab_values),
        "season": classify_season(lab_values),
        "face_shape": face_shape,
    }


__all__ = ["analyze_image_from_r2"]
