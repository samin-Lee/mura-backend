def analyze(landmarks):
    # 1. 얼굴 길이 분석
    face_height = abs(landmarks[10].y - landmarks[152].y)
    face_width = abs(landmarks[234].x - landmarks[454].x)
    face_shape = "long" if face_height / face_width > 1.5 else "short"

    # 2. 옆 턱 골격 분석
    jaw_width = abs(landmarks[172].x - landmarks[397].x)
    jaw_prominence = "prominent" if jaw_width / face_width > 0.85 else "smooth"

    # 3. 옆 광대 분석
    cheekbone_width = abs(landmarks[127].x - landmarks[356].x)
    side_cheek_prominence = "prominent" if cheekbone_width > face_width * 0.95 else "hidden"

    # 4. 앞 광대 분석
    front_cheek_depth = landmarks[116].z
    front_cheek_presence = "present" if front_cheek_depth < -0.05 else "flat"

    return {
        "face_length": face_shape,
        "jawline": jaw_prominence,
        "side_cheekbone": side_cheek_prominence,
        "front_cheekbone": front_cheek_presence
    }
