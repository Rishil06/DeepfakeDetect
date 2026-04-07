"""
extract_video_frames.py — Video Dataset Preparation for Deepfake Training
==========================================================================

Extracts frames from a folder of real/fake videos and organises them into
a frame dataset with a VIDEO-LEVEL train/val split (prevents data leakage).

Why video-level split matters
------------------------------
Splitting by FRAME (random shuffle) lets adjacent frames from the same video
appear in both train and val sets — the model memorises the video instead of
learning to detect deepfakes. A VIDEO-level split is the correct approach:
all frames from a given video go to either train or val, never both.

Supported input formats
-----------------------
MP4, AVI, MOV, MKV, WEBM (anything OpenCV can decode)

Supported dataset layouts
--------------------------
1. Flat layout (simplest):
       videos/
           real/  ← real .mp4 files
           fake/  ← fake/deepfake .mp4 files

2. FaceForensics++ layout:
       FF++/
           original_sequences/actors/raw/videos/   ← real
           manipulated_sequences/Deepfakes/raw/videos/  ← fake
           manipulated_sequences/Face2Face/raw/videos/  ← fake
           ... (use --ff_layout flag)

3. DFDC layout:
       dfdc_train/
           train_part_XX/
               metadata.json
               *.mp4          ← labelled by metadata.json
       (use --dfdc_layout flag)

4. CelebDF-v2 layout:
       Celeb-DF-v2/
           Celeb-real/    Celeb-synthesis/    YouTube-real/
       (use --celebdf_layout flag)

Output structure
----------------
    frames_out/
        train/
            real/   ← real frames (JPG)
            fake/   ← fake frames (JPG)
        val/
            real/
            fake/

Usage examples
--------------
    # Basic flat layout
    python extract_video_frames.py --video_dir ./videos --out_dir ./frames

    # With face crop (recommended — focuses model on face region)
    python extract_video_frames.py --video_dir ./videos --out_dir ./frames --face_crop

    # FaceForensics++
    python extract_video_frames.py --video_dir ./FF++ --out_dir ./frames --ff_layout

    # DFDC
    python extract_video_frames.py --video_dir ./dfdc_train --out_dir ./frames --dfdc_layout

    # Adjust sampling
    python extract_video_frames.py --video_dir ./videos --out_dir ./frames \\
        --fps 2 --max_frames_per_video 60 --val_split 0.2
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SUPPORTED_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
JPEG_QUALITY    = 95


# ─────────────────────────────────────────────
# Face detection (optional)
# ─────────────────────────────────────────────
_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade


def crop_face(frame_bgr: np.ndarray, margin: float = 0.25) -> np.ndarray | None:
    """
    Detect the largest face in the frame and return a cropped + padded region.
    Returns None if no face found (caller should use the full frame as fallback).
    margin: fractional padding around the detected bounding box (default 25%).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None

    # Use the largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    H, W = frame_bgr.shape[:2]

    pad_x = int(w * margin)
    pad_y = int(h * margin)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


# ─────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────
def extract_frames(
    video_path: str,
    out_dir: str,
    video_id: str,
    target_fps: float = 1.0,
    max_frames: int = 60,
    face_crop: bool = False,
    face_crop_fallback: bool = True,
) -> int:
    """
    Extract up to `max_frames` frames from a video at `target_fps`.
    Saves as JPEG files named  {video_id}_{frame_idx:06d}.jpg.

    Returns the number of frames successfully saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step    = max(1, int(src_fps / target_fps))

    # Build list of frame indices to extract
    positions = list(range(0, total, step))
    if len(positions) > max_frames:
        idxs = np.linspace(0, len(positions) - 1, max_frames, dtype=int)
        positions = [positions[i] for i in idxs]

    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        out_frame = frame
        if face_crop:
            cropped = crop_face(frame)
            if cropped is not None:
                out_frame = cropped
            elif not face_crop_fallback:
                continue  # skip frames with no face

        fname = f"{video_id}_{pos:06d}.jpg"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, out_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

    cap.release()
    return saved


# ─────────────────────────────────────────────
# Dataset layout parsers
# ─────────────────────────────────────────────
def collect_flat(video_dir: str) -> dict[str, list[tuple[str, int]]]:
    """
    Flat layout: video_dir/real/*.mp4 and video_dir/fake/*.mp4
    Returns {"real": [(path, 0), ...], "fake": [(path, 1), ...]}
    """
    result = {"real": [], "fake": []}
    for label, cls_int in [("real", 0), ("fake", 1)]:
        folder = Path(video_dir) / label
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected folder: {folder}\n"
                f"Create videos/real/ and videos/fake/ with your video files."
            )
        for f in folder.iterdir():
            if f.suffix.lower() in SUPPORTED_EXTS:
                result[label].append((str(f), cls_int))
    return result


def collect_ff_plus_plus(video_dir: str) -> dict[str, list[tuple[str, int]]]:
    """
    FaceForensics++ layout. Collects from all manipulation types.
    """
    result = {"real": [], "fake": []}
    root = Path(video_dir)

    real_dirs = [
        root / "original_sequences" / "actors" / "raw" / "videos",
        root / "original_sequences" / "youtube" / "raw" / "videos",
    ]
    fake_dirs = [
        root / "manipulated_sequences" / "Deepfakes" / "raw" / "videos",
        root / "manipulated_sequences" / "Face2Face" / "raw" / "videos",
        root / "manipulated_sequences" / "FaceShifter" / "raw" / "videos",
        root / "manipulated_sequences" / "FaceSwap" / "raw" / "videos",
        root / "manipulated_sequences" / "NeuralTextures" / "raw" / "videos",
    ]

    for d in real_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result["real"].append((str(f), 0))

    for d in fake_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result["fake"].append((str(f), 1))

    return result


def collect_dfdc(video_dir: str) -> dict[str, list[tuple[str, int]]]:
    """
    DFDC layout — reads metadata.json in each part folder for labels.
    label == "FAKE" in metadata → fake; otherwise real.
    """
    result = {"real": [], "fake": []}
    root = Path(video_dir)
    parts = sorted(p for p in root.iterdir() if p.is_dir())

    for part in parts:
        meta_path = part / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            metadata = json.load(f)
        for fname, info in metadata.items():
            vid_path = part / fname
            if not vid_path.exists():
                continue
            is_fake = info.get("label", "").upper() == "FAKE"
            cls = "fake" if is_fake else "real"
            result[cls].append((str(vid_path), 1 if is_fake else 0))

    return result


def collect_celebdf(video_dir: str) -> dict[str, list[tuple[str, int]]]:
    """CelebDF-v2 layout."""
    result = {"real": [], "fake": []}
    root = Path(video_dir)

    real_dirs = ["Celeb-real", "YouTube-real"]
    fake_dirs = ["Celeb-synthesis"]

    for d in real_dirs:
        p = root / d
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result["real"].append((str(f), 0))

    for d in fake_dirs:
        p = root / d
        if p.exists():
            for f in p.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTS:
                    result["fake"].append((str(f), 1))

    return result


# ─────────────────────────────────────────────
# Video-level train/val split
# ─────────────────────────────────────────────
def video_level_split(
    videos: list[tuple[str, int]],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Split a list of (video_path, label) tuples into train/val at the VIDEO level.
    All frames from a video will only appear in one split — no data leakage.
    """
    rng = random.Random(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    return shuffled[n_val:], shuffled[:n_val]


# ─────────────────────────────────────────────
# Main extraction runner
# ─────────────────────────────────────────────
def run_extraction(args):
    print(f"\n{'='*60}")
    print("  DeepfakeDetect — Video Frame Extractor")
    print(f"{'='*60}\n")

    # Collect video paths based on dataset layout
    if args.ff_layout:
        print("📂 Layout: FaceForensics++")
        videos_by_class = collect_ff_plus_plus(args.video_dir)
    elif args.dfdc_layout:
        print("📂 Layout: DFDC")
        videos_by_class = collect_dfdc(args.video_dir)
    elif args.celebdf_layout:
        print("📂 Layout: CelebDF-v2")
        videos_by_class = collect_celebdf(args.video_dir)
    else:
        print("📂 Layout: Flat (videos/real/ + videos/fake/)")
        videos_by_class = collect_flat(args.video_dir)

    all_real = videos_by_class["real"]
    all_fake = videos_by_class["fake"]
    print(f"  Found {len(all_real)} real videos, {len(all_fake)} fake videos")

    if not all_real or not all_fake:
        print("\n[ERROR] No videos found. Check your --video_dir path and layout flag.")
        return

    # Balance classes if requested
    if args.balance:
        n = min(len(all_real), len(all_fake))
        all_real = random.sample(all_real, n)
        all_fake = random.sample(all_fake, n)
        print(f"  Balanced to {n} per class")

    # Video-level split (prevents leakage)
    train_real, val_real = video_level_split(all_real, args.val_split, args.seed)
    train_fake, val_fake = video_level_split(all_fake, args.val_split, args.seed)

    splits = {
        "train": {"real": train_real, "fake": train_fake},
        "val":   {"real": val_real,   "fake": val_fake},
    }

    print(f"\n  Split (video-level, no leakage):")
    for split, classes in splits.items():
        for cls, vids in classes.items():
            print(f"    {split}/{cls}: {len(vids)} videos")

    # Extract frames
    total_saved = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}

    for split, classes in splits.items():
        for cls, videos in classes.items():
            out_dir = os.path.join(args.out_dir, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n  Extracting {split}/{cls} ({len(videos)} videos) …")

            for vid_path, _ in tqdm(videos, desc=f"  {split}/{cls}"):
                vid_id = Path(vid_path).stem
                n = extract_frames(
                    video_path=vid_path,
                    out_dir=out_dir,
                    video_id=vid_id,
                    target_fps=args.fps,
                    max_frames=args.max_frames_per_video,
                    face_crop=args.face_crop,
                    face_crop_fallback=not args.face_crop_strict,
                )
                total_saved[split][cls] += n

    # Summary
    print(f"\n{'='*60}")
    print("  ✅ Extraction complete")
    print(f"{'='*60}")
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            n = total_saved[split][cls]
            print(f"  {split}/{cls}: {n:,} frames")
    print(f"\n  Output: {args.out_dir}/")
    print("\n  Next step:")
    print(f"    python train_video.py --data_dir {args.out_dir} --epochs 10")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from real/fake videos for deepfake model training."
    )

    # Paths
    parser.add_argument("--video_dir", required=True,
        help="Root directory of your video dataset")
    parser.add_argument("--out_dir", default="./video_frames",
        help="Output directory for extracted frames (default: ./video_frames)")

    # Dataset layout
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--ff_layout", action="store_true",
        help="Use FaceForensics++ directory layout")
    layout.add_argument("--dfdc_layout", action="store_true",
        help="Use DFDC directory layout (reads metadata.json)")
    layout.add_argument("--celebdf_layout", action="store_true",
        help="Use CelebDF-v2 directory layout")

    # Sampling
    parser.add_argument("--fps", type=float, default=1.0,
        help="Frames to extract per second of video (default: 1.0)")
    parser.add_argument("--max_frames_per_video", type=int, default=60,
        help="Maximum frames to extract per video (default: 60)")

    # Face crop
    parser.add_argument("--face_crop", action="store_true",
        help="Crop to face region before saving (recommended for face-swap deepfakes)")
    parser.add_argument("--face_crop_strict", action="store_true",
        help="Only save frames where a face was detected (skip no-face frames)")

    # Split
    parser.add_argument("--val_split", type=float, default=0.2,
        help="Fraction of VIDEOS (not frames) to use for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true",
        help="Balance real/fake video counts by downsampling the majority class")

    args = parser.parse_args()
    run_extraction(args)
