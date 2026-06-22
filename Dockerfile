FROM python:3.11.9

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --only-binary=onnx --upgrade -r requirements.txt

COPY --chown=user . /app

RUN python -c "from analysis.face_landmarks import get_face_landmarker_model_path; get_face_landmarker_model_path()" \
    && python -c "from analysis.face_shape.model_loader import get_face_parsing_model_path; get_face_parsing_model_path()" \
    && python -c "from analysis.eyes.eye_metrics import get_insightface_app; get_insightface_app()"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
