from analysis.image_analyzer import decode_image, extract_skin_region
from analysis.lab_analyzer import calculate_lab_average
from analysis.personal_color_analyzer import (
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

    return {
        "lab_values": lab_values,
        "tone": classify_skin_tone(lab_values),
        "brightness": classify_brightness(lab_values),
        "season": classify_season(lab_values),
    }
