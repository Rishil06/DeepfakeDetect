"""
main.py — FastAPI backend for Deepfake Detection
Run: uvicorn main:app --reload --port 8000
Then open: http://localhost:8000
"""
import base64
import io
import os

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import torch

from predict import predict
from video_predict import predict_video

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FRONTEND_PATH = os.environ.get(
    "FRONTEND_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.html"),
)

ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/avi", "video/x-msvideo",
    "video/quicktime", "video/x-matroska", "video/webm",
    "video/x-ms-wmv", "video/mpeg",
}

VIDEO_EXT_MAP = {
    "video/mp4":        ".mp4",
    "video/avi":        ".avi",
    "video/x-msvideo":  ".avi",
    "video/quicktime":  ".mov",
    "video/x-matroska": ".mkv",
    "video/webm":       ".webm",
    "video/x-ms-wmv":   ".wmv",
    "video/mpeg":       ".mpeg",
}

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    description="Upload an image or video to detect if it is real or AI-generated.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fake_prob: float
    heatmap_b64: str | None
    models_used: list[str]


class FrameResultResponse(BaseModel):
    frame_index: int
    timestamp_sec: float
    label: str
    confidence: float
    fake_prob: float
    thumbnail_b64: str


class VideoPredictionResponse(BaseModel):
    label: str
    confidence: float
    fake_prob: float
    fake_frame_count: int
    total_frames_analysed: int
    video_duration_sec: float
    frames: list[FrameResultResponse]
    models_used: list[str]
    thumbnail_b64: str | None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_frontend():
    if not os.path.exists(FRONTEND_PATH):
        raise HTTPException(status_code=404, detail=f"Frontend not found at {FRONTEND_PATH}.")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "note": "Models are downloaded on-demand from HuggingFace on first inference.",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    heatmap: bool = Form(True),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    try:
        result = predict(image, generate_heatmap=heatmap)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    heatmap_b64 = None
    if result.get("heatmap_png"):
        heatmap_b64 = base64.b64encode(result["heatmap_png"]).decode("utf-8")

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        fake_prob=result["fake_prob"],
        heatmap_b64=heatmap_b64,
        models_used=result.get("models_used", []),
    )


@app.post("/predict/video", response_model=VideoPredictionResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    max_frames: int = Form(30),
):
    """
    Analyse a video for deepfake content.
    Supported: MP4, AVI, MOV, MKV, WEBM.
    Samples up to max_frames frames (default 30, max 60).
    """
    ct = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".mpeg", ".mpg"}
    ext = os.path.splitext(filename)[1]

    if ct not in ALLOWED_VIDEO_TYPES and ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail="Only video files are accepted. Supported: MP4, AVI, MOV, MKV, WEBM.",
        )

    max_frames = min(max(1, max_frames), 60)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_ext = VIDEO_EXT_MAP.get(ct, ext or ".mp4")

    try:
        result = predict_video(
            video_bytes=contents,
            file_ext=file_ext,
            max_frames=max_frames,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")

    return VideoPredictionResponse(
        label=result.label,
        confidence=result.confidence,
        fake_prob=result.fake_prob,
        fake_frame_count=result.fake_frame_count,
        total_frames_analysed=result.total_frames_analysed,
        video_duration_sec=result.video_duration_sec,
        frames=[
            FrameResultResponse(
                frame_index=f.frame_index,
                timestamp_sec=f.timestamp_sec,
                label=f.label,
                confidence=f.confidence,
                fake_prob=f.fake_prob,
                thumbnail_b64=f.thumbnail_b64,
            )
            for f in result.frames
        ],
        models_used=result.models_used,
        thumbnail_b64=result.thumbnail_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
