import cv2


def extract_skin_region(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, width, height = faces[0]
        face = image[y : y + height, x : x + width]
    else:
        face = image

    height, width, _ = face.shape
    return face[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4]
