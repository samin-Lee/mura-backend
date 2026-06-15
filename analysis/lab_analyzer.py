import cv2


def calculate_lab_average(image_region) -> dict:
    lab = cv2.cvtColor(image_region, cv2.COLOR_BGR2LAB)
    avg_color = lab.mean(axis=0).mean(axis=0)

    return {
        "L": int(avg_color[0]),
        "A": int(avg_color[1]),
        "B": int(avg_color[2]),
    }