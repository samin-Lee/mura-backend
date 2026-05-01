import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def get_face_landmarks(img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark