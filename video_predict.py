"""
video_predict.py — Deepfake Detection for Video Files
Extracts frames from a video, runs inference on each sampled frame,
and aggregates results into a per-frame timeline + final verdict.

Inference mode (auto-selected):
  1. Custom fine-tuned model  — if MODEL_PATH env var points to a .pth checkpoint
     trained with train_video.py, that model is used for all frame inference.
     This gives the best accuracy for deepfake videos.
  2. HuggingFace ensemble     — the default 3-model ensemble from predict.py.
     No local checkpoint required; models download on first use.

Supported video formats: MP4, AVI, MOV, MKV, WEBM (anything OpenCV can decode)
"""

import io
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from predict import predict, load_pipelines


# ─────────────────────────────────────────────
# Custom model loading (trained with train_video.py)
# ─────────────────────────────────────────────
_custom_model_cache: dict = {}


def _load_custom_model(model_path: str):
    """
    Load a fine-tuned EfficientNet checkpoint saved by train_video.py or train.py.
    Cached after first load — subsequent calls return the cached model.
    """
    if "model" in _custom_model_cache:
        return _custom_model_cache["model"], _custom_model_cache["transform"]

    import timm

    print(f"[INFO] Loading custom video model from {model_path} …")
    ckpt = torch.load(model_path, map_location="cpu")

    model_name = ckpt.get("model_name", "efficientnet_b4")
    img_size   = ckpt.get("img_size", 224)

    import torch.nn as nn
    model = timm.create_model(model_name, pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 1),
    )
    model.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_auc = ckpt.get("val_auc", "?")
    trained_on = ckpt.get("trained_on", "images")
    print(f"[INFO]   ✅ Custom model ready  "
          f"({model_name}, img_size={img_size}, "
          f"trained_on={trained_on}, val_auc={val_auc})")

    _custom_model_cache["model"]     = model
    _custom_model_cache["transform"] = tf
    _custom_model_cache["device"]    = device
    return model, tf


@torch.no_grad()
def _predict_with_custom_model(pil_image: Image.Image, model_path: str) -> dict:
    """Run inference with the custom fine-tuned model on a single frame."""
    model, tf = _load_custom_model(model_path)
    device = _custom_model_cache["device"]

    tensor = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)
    logit  = model(tensor)
    prob   = torch.sigmoid(logit).item()

    # Apply calibration sigmoid (same as ensemble pipeline)
    x = (prob - 0.5) * 6.0
    fake_prob = 1.0 / (1.0 + math.exp(-x))

    label = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = round(fake_prob * 100 if label == "FAKE" else (1 - fake_prob) * 100, 2)

    return {
        "label":      label,
        "confidence": confidence,
        "fake_prob":  round(fake_prob, 4),
        "models_used": [os.path.basename(model_path)],
    }


# ─────────────────────────────────────────────
# Config defaults
# ─────────────────────────────────────────────
MAX_FRAMES          = 30    # maximum frames to sample from any video
SAMPLE_INTERVAL_SEC = 1.0   # try to sample 1 frame per second
MIN_FRAMES          = 3     # analyse at least this many frames


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class FrameResult:
    frame_index: int          # original 0-based frame number
    timestamp_sec: float      # position in the video (seconds)
    label: str                # "REAL" or "FAKE"
    confidence: float         # 0–100
    fake_prob: float          # 0–1 calibrated ensemble probability
    thumbnail_b64: str        # base64 JPEG thumbnail for UI


@dataclass
class VideoResult:
    label: str                       # overall verdict
    confidence: float                # 0–100
    fake_prob: float                 # ensemble average across frames
    fake_frame_count: int            # how many frames were classified FAKE
    total_frames_analysed: int       # frames actually processed
    video_duration_sec: float        # total video length
    frames: list[FrameResult]        # per-frame details
    models_used: list[str]           # HuggingFace models that ran
    thumbnail_b64: Optional[str]     # first frame thumbnail


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _frame_to_b64_jpeg(frame_bgr: np.ndarray, size: int = 160) -> str:
    """Resize an OpenCV BGR frame → base64 JPEG string."""
    import base64
    h, w = frame_bgr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _bgr_frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pick_frame_positions(
    total_frames: int,
    fps: float,
    max_frames: int = MAX_FRAMES,
    interval_sec: float = SAMPLE_INTERVAL_SEC,
) -> list[int]:
    """
    Return a list of 0-based frame indices to sample.
    Strategy: 1 frame per interval_sec, capped at max_frames,
    always including the first and last frame.
    """
    if total_frames <= 0 or fps <= 0:
        return []

    step = max(1, int(fps * interval_sec))
    positions = list(range(0, total_frames, step))

    # Always include last frame
    if positions[-1] != total_frames - 1:
        positions.append(total_frames - 1)

    # Cap to max_frames (uniform sub-sample if over)
    if len(positions) > max_frames:
        indices = np.linspace(0, len(positions) - 1, max_frames, dtype=int)
        positions = [positions[i] for i in indices]

    return sorted(set(positions))


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def predict_video(
    video_bytes: bytes,
    file_ext: str = ".mp4",
    max_frames: int = MAX_FRAMES,
    progress_callback=None,          # optional callable(current, total)
) -> VideoResult:
    """
    Analyse a video for deepfake content.

    Parameters
    ----------
    video_bytes      : raw video file bytes
    file_ext         : file extension hint (e.g. ".mp4", ".avi")
    max_frames       : maximum frames to analyse (default 30)
    progress_callback: optional fn(current_frame, total_frames) for SSE / logging

    Returns
    -------
    VideoResult dataclass with per-frame breakdown + overall verdict
    """

    # ── Determine inference mode ──────────────────────────────────────
    model_path = os.environ.get("MODEL_PATH", "").strip()
    use_custom = bool(model_path and os.path.isfile(model_path))

    if use_custom:
        print(f"[INFO] Video inference: custom model ({model_path})")
        _load_custom_model(model_path)   # pre-warm
    else:
        print("[INFO] Video inference: HuggingFace ensemble")
        load_pipelines()   # pre-warm ensemble

    # ── Write to temp file (OpenCV needs a real path) ──
    suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError(
                f"OpenCV could not open the video file. "
                f"Make sure the format is supported (MP4, AVI, MOV, MKV, WEBM)."
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        if total_frames < 1:
            raise ValueError("Video has no readable frames.")

        frame_positions = _pick_frame_positions(
            total_frames, fps, max_frames=max_frames
        )
        n_to_analyse = len(frame_positions)

        print(
            f"[INFO] Video: {total_frames} frames @ {fps:.1f}fps, "
            f"{duration_sec:.1f}s → analysing {n_to_analyse} frames"
        )

        frame_results: list[FrameResult] = []
        models_used_set: set[str] = set()
        first_thumb: Optional[str] = None

        for i, frame_idx in enumerate(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                print(f"[WARN] Could not read frame {frame_idx}, skipping.")
                continue

            timestamp = frame_idx / fps
            pil_image = _bgr_frame_to_pil(frame_bgr)
            thumb_b64 = _frame_to_b64_jpeg(frame_bgr)

            if first_thumb is None:
                first_thumb = thumb_b64

            # Run inference: custom model or HuggingFace ensemble
            try:
                if use_custom:
                    result = _predict_with_custom_model(pil_image, model_path)
                else:
                    result = predict(pil_image, generate_heatmap=False)
            except Exception as exc:
                print(f"[WARN] predict() failed on frame {frame_idx}: {exc}")
                continue

            for m in result.get("models_used", []):
                models_used_set.add(m)

            frame_results.append(
                FrameResult(
                    frame_index=frame_idx,
                    timestamp_sec=round(timestamp, 2),
                    label=result["label"],
                    confidence=result["confidence"],
                    fake_prob=result["fake_prob"],
                    thumbnail_b64=thumb_b64,
                )
            )

            if progress_callback:
                progress_callback(i + 1, n_to_analyse)

            print(
                f"[DEBUG] Frame {frame_idx} ({timestamp:.2f}s): "
                f"{result['label']} @ {result['confidence']:.1f}%"
            )

        cap.release()

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not frame_results:
        raise RuntimeError(
            "No frames could be successfully analysed. "
            "Check that the video is not corrupted."
        )

    # ── Aggregate ──
    avg_fake_prob = sum(f.fake_prob for f in frame_results) / len(frame_results)
    fake_count    = sum(1 for f in frame_results if f.label == "FAKE")

    # Use weighted aggregation: bias toward extreme frames
    # (a video where 60%+ of frames are FAKE is conclusively fake)
    fake_ratio    = fake_count / len(frame_results)

    # Sigmoid-sharpen the ratio for cleaner decisions
    x = (fake_ratio - 0.5) * 6.0
    sharpened_ratio = 1.0 / (1.0 + math.exp(-x))

    # Blend ensemble probability with frame-vote ratio
    final_fake_prob = 0.6 * avg_fake_prob + 0.4 * sharpened_ratio

    if final_fake_prob >= 0.5:
        label      = "FAKE"
        confidence = round(final_fake_prob * 100, 2)
    else:
        label      = "REAL"
        confidence = round((1.0 - final_fake_prob) * 100, 2)

    return VideoResult(
        label=label,
        confidence=confidence,
        fake_prob=round(final_fake_prob, 4),
        fake_frame_count=fake_count,
        total_frames_analysed=len(frame_results),
        video_duration_sec=round(duration_sec, 2),
        frames=frame_results,
        models_used=sorted(models_used_set),
        thumbnail_b64=first_thumb,
    )
