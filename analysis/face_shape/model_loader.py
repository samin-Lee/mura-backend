FACE_PARSING_REPO_ID = "PayamFard123/dermaintel-face-parsing"
FACE_PARSING_FILENAME = "resnet18.onnx"


def get_face_parsing_model_path() -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    return hf_hub_download(
        repo_id=FACE_PARSING_REPO_ID,
        filename=FACE_PARSING_FILENAME,
    )
