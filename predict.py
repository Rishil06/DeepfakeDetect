"""
predict.py — Deepfake Detection Inference
Uses a 3-model HuggingFace ensemble (no local checkpoint required).
Models are downloaded automatically on first call and cached by HuggingFace.

Models used:
  1. umm-maybe/AI-image-detector   — general AI photos / GAN faces
  2. Organika/sdxl-detector        — SDXL / Midjourney / AI art
  3. cmckinle/sdxl-flux-detector   — Flux / newer generators
"""

import io
import math

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


# ─────────────────────────────────────────────
# Model registry  (name, ensemble weight)
# ─────────────────────────────────────────────
MODELS = [
    ("umm-maybe/AI-image-detector", 0.40),
    ("Organika/sdxl-detector",      0.35),
    ("cmckinle/sdxl-flux-detector", 0.25),
]

# Module-level cache so pipelines are only loaded once per process
_pipeline_cache: dict = {}


# ─────────────────────────────────────────────
# Pipeline loading
# ─────────────────────────────────────────────
def load_pipelines() -> list:
    """
    Load (or return cached) HuggingFace image-classification pipelines.
    Downloads models on first call; subsequent calls use the in-memory cache.
    """
    if "pipes" in _pipeline_cache:
        return _pipeline_cache["pipes"]

    device = 0 if torch.cuda.is_available() else -1
    pipes = []

    for model_name, weight in MODELS:
        try:
            print(f"[INFO] Loading {model_name} …")
            pipe = pipeline(
                "image-classification",
                model=model_name,
                device=device,
            )
            pipes.append((model_name, pipe, weight))
            print(f"[INFO]   ✅ {model_name} ready")
        except Exception as exc:
            print(f"[WARN]   ⚠️  Skipping {model_name}: {exc}")

    if not pipes:
        raise RuntimeError(
            "No classification models could be loaded. "
            "Check your internet connection and HuggingFace access."
        )

    _pipeline_cache["pipes"] = pipes
    return pipes


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _get_fake_prob(pipe_result: list) -> float:
    """
    Robustly extract the 'fake / AI-generated' probability from any
    model's output regardless of its label naming convention.
    """
    label_map = {r["label"].lower(): r["score"] for r in pipe_result}

    # Exact-match keys that mean AI/fake
    fake_keys = [
        "artificial", "fake", "ai", "generated", "sd",
        "midjourney", "flux", "sdxl", "diffusion", "synthetic",
    ]
    for key in fake_keys:
        if key in label_map:
            return label_map[key]

    # Partial-match fallback
    for label, score in label_map.items():
        for fk in fake_keys:
            if fk in label:
                return score

    # Last resort: lowest score is assumed to be fake
    # (most models score the "real" class highest)
    sorted_items = sorted(pipe_result, key=lambda x: x["score"])
    return sorted_items[0]["score"]


def _calibrate(prob: float) -> float:
    """
    Sharpen probabilities away from the 0.5 decision boundary using a
    scaled sigmoid so uncertain scores are nudged toward a cleaner decision.
    """
    x = (prob - 0.5) * 6.0
    return 1.0 / (1.0 + math.exp(-x))


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def predict(
    image: Image.Image,
    generate_heatmap: bool = True,
) -> dict:
    """
    Run the ensemble and return a result dict:
      - label        : "REAL" or "FAKE"
      - confidence   : float, 0–100
      - fake_prob    : float, 0–1  (calibrated ensemble probability)
      - heatmap_png  : bytes | None  (PNG image bytes)
      - models_used  : list[str]
    """
    pipes = load_pipelines()
    image_rgb = image.convert("RGB")

    weighted_sum = 0.0
    total_weight = 0.0
    models_used = []

    for model_name, pipe, weight in pipes:
        try:
            result = pipe(image_rgb)
            fp = _get_fake_prob(result)
            weighted_sum += fp * weight
            total_weight += weight
            models_used.append(model_name)
            print(f"[DEBUG] {model_name}: fake_prob={fp:.4f}  raw={result}")
        except Exception as exc:
            print(f"[WARN] {model_name} inference failed: {exc}")

    if total_weight == 0:
        fake_prob = 0.5
    else:
        raw_prob = weighted_sum / total_weight
        fake_prob = _calibrate(raw_prob)

    print(f"[DEBUG] Ensemble fake_prob (calibrated) = {fake_prob:.4f}")

    # Decision
    if fake_prob >= 0.5:
        label = "FAKE"
        confidence = round(fake_prob * 100, 2)
    else:
        label = "REAL"
        confidence = round((1.0 - fake_prob) * 100, 2)

    # Heatmap
    heatmap_png = None
    if generate_heatmap:
        try:
            heatmap_png = _frequency_heatmap(image_rgb)
        except Exception as exc:
            print(f"[WARN] FFT heatmap failed ({exc}), trying edge heatmap …")
            try:
                heatmap_png = _edge_heatmap(image_rgb)
            except Exception as exc2:
                print(f"[WARN] Edge heatmap also failed: {exc2}")

    return {
        "label": label,
        "confidence": confidence,
        "fake_prob": round(fake_prob, 4),
        "heatmap_png": heatmap_png,
        "models_used": models_used,
    }


# ─────────────────────────────────────────────
# Heatmap generators
# ─────────────────────────────────────────────
def _frequency_heatmap(image: Image.Image, size: int = 224) -> bytes:
    """
    FFT-based anomaly heatmap.
    AI-generated images have unnatural frequency patterns that stand out
    clearly in the magnitude spectrum — this overlays them on the source image.
    """
    img_np = np.array(image.resize((size, size)).convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1.0)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = cv2.resize(magnitude, (size, size))

    heatmap_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_INFERNO)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_np + 0.45 * heatmap_colored).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def _edge_heatmap(image: Image.Image, size: int = 224) -> bytes:
    """Fallback: Canny edge-detection overlay."""
    img_np = np.array(image.resize((size, size)).convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    edges_colored = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2RGB)
    overlay = (0.60 * img_np + 0.40 * edges_colored).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
