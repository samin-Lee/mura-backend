from threading import Lock
from typing import Optional, Sequence

import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark


class MediaPipeFaceMeshLoader:
    # MediaPipe FaceMesh는 초기화 비용이 크므로 애플리케이션 전체에서 하나만 생성합니다.
    _instance: Optional["MediaPipeFaceMeshLoader"] = None
    _instance_lock = Lock()

    def __new__(cls) -> "MediaPipeFaceMeshLoader":
        # 여러 요청이 동시에 들어와도 FaceMesh 인스턴스가 중복 생성되지 않도록 잠급니다.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        # FaceMesh의 process 호출은 내부 상태를 사용하므로 별도 락으로 요청 간 충돌을 막습니다.
        self._process_lock = Lock()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_face_landmarks(self, image_bgr) -> Optional[Sequence[NormalizedLandmark]]:
        # OpenCV는 BGR을 사용하고 MediaPipe는 RGB를 기대하므로 색상 채널을 변환합니다.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        with self._process_lock:
            results = self._face_mesh.process(image_rgb)
        # 얼굴이 없으면 API 계층에서 422 응답으로 변환할 수 있도록 None을 반환합니다.
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark


def get_face_landmarks(image_bgr) -> Optional[Sequence[NormalizedLandmark]]:
    # 외부 모듈은 싱글턴 클래스를 직접 다루지 않고 이 함수만 호출하면 됩니다.
    return MediaPipeFaceMeshLoader().get_face_landmarks(image_bgr)
