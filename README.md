# 🔍 DeepfakeDetect

Binary classifier that distinguishes **real / authentic content** from **AI-generated / deepfake images and videos** using a 3-model HuggingFace ensemble with FFT frequency heatmap visualisation.

> **No pre-trained checkpoint download required.** The 3 HuggingFace models are fetched automatically on the first inference call and cached locally by the `transformers` library.

---

## Project Structure

```
deepfake_detect/
├── main.py               ← FastAPI backend  (entry point)
├── predict.py            ← 3-model HuggingFace ensemble inference (images)
├── video_predict.py      ← Video frame sampler + ensemble aggregator
├── frontend.html         ← Web UI with Image + Video tabs (auto-served at /)
├── train.py              ← (optional) train a custom EfficientNet-B4
├── download_dataset.py   ← (optional) download Kaggle dataset
├── train_colab.ipynb     ← (optional) Google Colab training notebook
└── requirements.txt
```

---

## Quick Start (2 steps)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Launch the app

```bash
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

The three HuggingFace models download automatically on the **first upload** (one-time, ~500MB total). Subsequent runs use the local cache.

---

## Features

### Image Detection
Upload any image (JPG, PNG, WEBP, GIF) and get an instant verdict with:
- REAL / FAKE label + confidence score
- FFT frequency heatmap overlay showing anomaly patterns
- Per-model breakdown (3-model ensemble)

### Video Detection _(new)_
Upload a video (MP4, AVI, MOV, MKV, WEBM) and get:
- Overall REAL / FAKE verdict + confidence
- Frame-by-frame fake probability timeline chart
- Scrollable frame strip with per-frame thumbnails coloured by verdict
- Fake frame count / total frames analysed / video duration
- Configurable frame sampling (15 / 30 / 60 frames)

---

## How It Works

### Ensemble models

| Model | Weight | Speciality |
|-------|--------|-----------|
| `umm-maybe/AI-image-detector` | 40% | GAN faces, photo-realistic AI images |
| `Organika/sdxl-detector` | 35% | SDXL / Midjourney / AI art |
| `cmckinle/sdxl-flux-detector` | 25% | Flux & next-gen generators |

### Video Analysis Pipeline

1. **Frame Sampling** — OpenCV extracts 1 frame/sec (configurable), capped at `max_frames`
2. **Per-Frame Inference** — the 3-model ensemble runs on each sampled frame
3. **Aggregation** — fake probabilities are averaged across frames; a frame-vote ratio is blended in (60/40 split) and sigmoid-sharpened for a clean final verdict
4. **Timeline** — per-frame results are returned for the chart and frame strip in the UI

### FFT Heatmap (images only)
The frequency heatmap overlays the image's FFT magnitude spectrum. AI-generated images show unnatural periodic patterns invisible to the naked eye but highlighted by this visualisation.

---

## REST API

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/`              | Serve the web UI |
| `GET`  | `/health`        | Liveness check |
| `POST` | `/predict`       | Upload image → get verdict |
| `POST` | `/predict/video` | Upload video → per-frame analysis |

### Image endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@photo.jpg" \
  -F "heatmap=true"
```

### Video endpoint

```bash
curl -X POST http://localhost:8000/predict/video \
  -F "file=@clip.mp4" \
  -F "max_frames=30"
```

**Video response:**
```json
{
  "label": "FAKE",
  "confidence": 87.4,
  "fake_prob": 0.874,
  "fake_frame_count": 22,
  "total_frames_analysed": 30,
  "video_duration_sec": 31.5,
  "models_used": ["AI-image-detector", "sdxl-detector", "sdxl-flux-detector"],
  "thumbnail_b64": "...",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "label": "FAKE",
      "confidence": 91.2,
      "fake_prob": 0.912,
      "thumbnail_b64": "..."
    }
  ]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRONTEND_PATH` | `./frontend.html` | Path to the UI HTML file |
| `MODEL_PATH` | *(not used by default)* | Custom checkpoint path (optional) |

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Models | HuggingFace `transformers`, `timm` |
| Backend | `FastAPI`, `uvicorn` |
| Image/Video processing | `OpenCV`, `Pillow`, `NumPy` |
| Deep learning | `PyTorch` |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |

---

## Training a Custom Video Deepfake Model

### Step 1 — Extract frames from your videos

```bash
# Basic flat layout (videos/real/*.mp4 and videos/fake/*.mp4)
python extract_video_frames.py --video_dir ./videos --out_dir ./video_frames

# With face crop (recommended for face-swap deepfakes)
python extract_video_frames.py --video_dir ./videos --out_dir ./video_frames --face_crop

# FaceForensics++ dataset
python extract_video_frames.py --video_dir ./FF++ --out_dir ./video_frames --ff_layout

# DFDC dataset
python extract_video_frames.py --video_dir ./dfdc_train --out_dir ./video_frames --dfdc_layout

# CelebDF-v2 dataset
python extract_video_frames.py --video_dir ./Celeb-DF-v2 --out_dir ./video_frames --celebdf_layout
```

The extractor applies a **video-level train/val split** (not frame-level) to prevent data leakage.
All frames from any given video will only appear in one split.

### Step 2 — Train

```bash
python train_video.py --data_dir ./video_frames --epochs 15
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--tc_loss_weight` | 0.1 | Temporal consistency loss weight (0 to disable) |
| `--freeze_epochs` | 2 | Warm up with frozen backbone before full fine-tuning |
| `--balance_sampler` | off | Balance real/fake frames per batch |
| `--face_crop` | off | Applied at extraction time, not training time |

### Step 3 — Use your model

```bash
MODEL_PATH=./models/best_video_model.pth uvicorn main:app --port 8000
```

When `MODEL_PATH` is set, video analysis (`/predict/video`) uses your fine-tuned model.
Image analysis (`/predict`) continues to use the HuggingFace ensemble.

### Colab training

Open `train_video_colab.ipynb` in Google Colab for GPU-accelerated training (free T4).

### Evaluation metrics

Training outputs two AUC scores:
- **Frame AUC** — AUC over individual frames (inflated, not realistic)
- **Video AUC** — frames aggregated per-video, then AUC computed (realistic deployment metric)

Always optimise for **Video AUC**.
