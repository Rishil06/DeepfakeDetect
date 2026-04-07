"""
train_video.py — Fine-tune EfficientNet-B4 on Video Frames for Deepfake Detection
===================================================================================

Key differences from train.py (image training):

1.  VIDEO-LEVEL evaluation  — validation groups frames by video identity and
    aggregates per-video scores. This gives realistic accuracy numbers instead
    of inflated per-frame metrics.

2.  Video-specific augmentations — JPEG compression artifacts, temporal blur,
    Gaussian noise, and block artifacts that simulate real-world codec degradation.

3.  Temporal consistency loss (optional)  — frames extracted from the same video
    in the same batch are expected to have similar predictions. A consistency
    penalty discourages the model from flip-flopping on adjacent frames.

4.  Checkpoint compatibility — the saved model uses the same format as train.py
    so it can be loaded directly by predict.py and video_predict.py.

Prerequisites
-------------
Run extract_video_frames.py first to prepare the frame dataset:
    python extract_video_frames.py --video_dir ./videos --out_dir ./video_frames

Then train:
    python train_video.py --data_dir ./video_frames --epochs 15

Use your custom model:
    MODEL_PATH=./models/best_video_model.pth uvicorn main:app --port 8000
"""

import argparse
import copy
import os
import random
import re
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────
# Video-specific augmentations
# ─────────────────────────────────────────────
class JPEGCompression:
    """
    Simulate JPEG compression artifacts common in shared/re-encoded videos.
    The model must learn to detect deepfakes even through compression noise.
    """
    def __init__(self, quality_range=(40, 95)):
        self.low, self.high = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.low, self.high)
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class GaussianNoise:
    """Add Gaussian noise to simulate sensor noise and re-encoding artifacts."""
    def __init__(self, std_range=(0.005, 0.03)):
        self.low, self.high = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.low, self.high)
        return (tensor + torch.randn_like(tensor) * std).clamp(0, 1)


class BlockArtifacts:
    """
    Simulate block artifacts from low-bitrate video encoding.
    Applies a subtle grid pattern to random 8x8 blocks.
    """
    def __init__(self, p=0.3, intensity_range=(0.01, 0.06)):
        self.p = p
        self.low, self.high = intensity_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        intensity = random.uniform(self.low, self.high)
        h, w = arr.shape[:2]
        for y in range(0, h, 8):
            arr[y:y+1, :] = np.clip(arr[y:y+1, :] + intensity * 255, 0, 255)
        for x in range(0, w, 8):
            arr[:, x:x+1] = np.clip(arr[:, x:x+1] + intensity * 255, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))


class MotionBlur:
    """Horizontal motion blur to simulate camera movement."""
    def __init__(self, p=0.2, kernel_range=(3, 9)):
        self.p = p
        self.low, self.high = kernel_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.randrange(self.low, self.high, 2)
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        arr = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(arr)


def get_video_transforms(img_size: int = 224):
    """
    Training transforms with video-specific augmentations applied
    in this order: geometry → video degradation → tensor → noise.
    Validation uses only resize + normalize (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                   saturation=0.15, hue=0.05)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.05),
        JPEGCompression(quality_range=(45, 95)),
        BlockArtifacts(p=0.25),
        MotionBlur(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomApply([GaussianNoise(std_range=(0.005, 0.025))], p=0.4),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
_VIDEO_ID_RE = re.compile(r"^(.+?)_\d{6}\.jpg$")


def _parse_video_id(filename: str) -> str:
    """
    Extract video identity from frame filename.
    Frames are named {video_id}_{frame_idx:06d}.jpg by extract_video_frames.py.
    Falls back to the full stem if the pattern doesn't match.
    """
    m = _VIDEO_ID_RE.match(filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


class VideoFrameDataset(Dataset):
    """
    Dataset of extracted video frames.

    Loads frames from:
        data_dir/
            real/  ← JPG frames from real videos
            fake/  ← JPG frames from deepfake videos

    Each sample returns (image_tensor, label, video_id) so that the
    per-video aggregation in validation can group frames correctly.
    """

    def __init__(self, data_dir: str, transform=None):
        self.samples: list[tuple[str, int, str]] = []  # (path, label, video_id)
        self.transform = transform

        for label_int, folder in [(0, "real"), (1, "fake")]:
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"Expected folder: {folder_path}\n"
                    "Run extract_video_frames.py first."
                )
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(folder_path, fname)
                    vid_id = _parse_video_id(fname)
                    self.samples.append((path, label_int, vid_id))

        real_count = sum(1 for _, l, _ in self.samples if l == 0)
        fake_count = sum(1 for _, l, _ in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} frames from {data_dir} "
              f"(real: {real_count}, fake: {fake_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, video_id = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, video_id

    def class_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler (class balancing)."""
        counts = [0, 0]
        for _, l, _ in self.samples:
            counts[l] += 1
        total = sum(counts)
        weights_per_class = [total / (2 * c) for c in counts]
        return [weights_per_class[l] for _, l, _ in self.samples]


# Custom collate to handle the extra video_id string
def collate_fn(batch):
    images  = torch.stack([b[0] for b in batch])
    labels  = torch.tensor([b[1] for b in batch])
    vid_ids = [b[2] for b in batch]
    return images, labels, vid_ids


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def build_model(
    model_name: str = "efficientnet_b4",
    pretrained: bool = True,
    dropout: float = 0.4,
    freeze_backbone_epochs: int = 0,
) -> nn.Module:
    """
    EfficientNet-B4 with a binary classification head.
    Compatible with the checkpoint format used by predict.py.
    """
    model = timm.create_model(model_name, pretrained=pretrained, drop_rate=dropout)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────
# Temporal consistency loss
# ─────────────────────────────────────────────
class TemporalConsistencyLoss(nn.Module):
    """
    Penalises large prediction differences between frames from the same video.

    Rationale: a real face should look consistently real across all frames;
    an AI-generated face should look consistently fake. Large flip-flops on
    adjacent frames are usually wrong — this loss regularises against them.

    Only applied when a batch contains multiple frames from the same video.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        logits: torch.Tensor,      # (B, 1)
        video_ids: list[str],
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits).squeeze(1)   # (B,)
        device = probs.device

        # Group indices by video
        vid_groups: dict[str, list[int]] = defaultdict(list)
        for i, vid_id in enumerate(video_ids):
            vid_groups[vid_id].append(i)

        losses = []
        for indices in vid_groups.values():
            if len(indices) < 2:
                continue
            group_probs = probs[indices]
            # MSE between all pairs in the group
            diff = group_probs.unsqueeze(0) - group_probs.unsqueeze(1)
            losses.append((diff ** 2).mean())

        if not losses:
            return torch.tensor(0.0, device=device)

        return self.weight * torch.stack(losses).mean()


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, tc_loss_fn,
    optimizer, device, scaler,
    use_tc_loss: bool = True,
):
    model.train()
    running_loss = running_correct = running_total = 0

    for imgs, labels, vid_ids in tqdm(loader, desc="  Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            bce    = criterion(logits, labels)
            tc     = tc_loss_fn(logits, vid_ids) if use_tc_loss else 0.0
            loss   = bce + tc

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss    += loss.item() * imgs.size(0)
        preds            = (torch.sigmoid(logits) > 0.5).long()
        running_correct += (preds == labels.long()).sum().item()
        running_total   += imgs.size(0)

    return running_loss / running_total, running_correct / running_total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate on the validation set with TWO levels of metrics:
    1. Per-frame  — standard accuracy / AUC over individual frames
    2. Per-video  — aggregate per-video prediction, then compute AUC
                    (this is the realistic metric for deployment)
    """
    model.eval()
    running_loss = correct = total = 0

    all_probs:  list[float] = []
    all_labels: list[int]   = []

    # For per-video aggregation
    video_probs:  dict[str, list[float]] = defaultdict(list)
    video_labels: dict[str, int]         = {}

    for imgs, labels, vid_ids in tqdm(loader, desc="  Val  ", leave=False):
        imgs   = imgs.to(device)
        labels_t = labels.float().unsqueeze(1).to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels_t)
        probs  = torch.sigmoid(logits).squeeze(1)

        running_loss += loss.item() * imgs.size(0)
        preds         = (probs > 0.5).long()
        correct      += (preds == labels.to(device)).sum().item()
        total        += imgs.size(0)

        probs_np = probs.cpu().numpy().tolist()
        for prob, label, vid_id in zip(probs_np, labels.tolist(), vid_ids):
            all_probs.append(prob)
            all_labels.append(label)
            video_probs[vid_id].append(prob)
            video_labels[vid_id] = label

    # Per-frame AUC
    frame_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5

    # Per-video AUC (mean-aggregate per video then score)
    vid_avg_probs  = [np.mean(video_probs[v]) for v in video_probs]
    vid_true_labels = [video_labels[v] for v in video_probs]
    video_auc = (
        roc_auc_score(vid_true_labels, vid_avg_probs)
        if len(set(vid_true_labels)) > 1 else 0.5
    )

    return (
        running_loss / total,
        correct / total,
        frame_auc,
        video_auc,
        len(video_probs),   # number of unique videos in val set
    )


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────
def save_checkpoint(model, path, model_name, val_auc, val_acc, img_size):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "val_auc":    val_auc,
        "val_acc":    val_acc,
        "img_size":   img_size,
        "trained_on": "video_frames",
    }, path)
    print(f"  ✅ Checkpoint saved → {path}  (AUC={val_auc:.4f})")


# ─────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────
def plot_history(history: dict, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(epochs, history["frame_auc"], label="Frame AUC")
    axes[2].plot(epochs, history["video_auc"], label="Video AUC")
    axes[2].set_title("AUC (Frame vs Video)"); axes[2].legend()

    axes[3].plot(epochs, history["lr"])
    axes[3].set_title("Learning Rate")

    plt.tight_layout()
    out = os.path.join(save_dir, "video_training_curves.png")
    plt.savefig(out, dpi=130)
    print(f"  📊 Training curves → {out}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  DeepfakeDetect — Video Frame Training")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────
    train_tf, val_tf = get_video_transforms(img_size=args.img_size)

    print("Loading datasets …")
    train_ds = VideoFrameDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_ds   = VideoFrameDataset(os.path.join(args.data_dir, "val"),   val_tf)

    # Class-balanced sampler (handles real/fake imbalance)
    if args.balance_sampler:
        weights  = train_ds.class_weights()
        sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle  = False
    else:
        sampler  = None
        shuffle  = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────
    print(f"\nBuilding {args.model_name} …")
    model = build_model(args.model_name, pretrained=True, dropout=args.dropout)
    model = model.to(device)

    # Optional: warm up with frozen backbone first
    if args.freeze_epochs > 0:
        print(f"  Freezing backbone for first {args.freeze_epochs} epoch(s) …")
        freeze_backbone(model)

    # ── Loss / optimiser / scheduler ──────────────
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tc_loss_fn = TemporalConsistencyLoss(weight=args.tc_loss_weight)
    optimizer  = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Training ──────────────────────────────────
    best_auc  = 0.0
    history   = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "frame_auc":  [], "video_auc": [],
        "lr": [],
    }
    save_path = os.path.join(args.model_dir, "best_video_model.pth")

    print(f"\n🚀 Training for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            print(f"\n  [Epoch {epoch}] Unfreezing full backbone …")
            unfreeze_all(model)
            # Re-create optimizer to include all params
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        use_tc = args.tc_loss_weight > 0
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, tc_loss_fn,
            optimizer, device, scaler, use_tc_loss=use_tc,
        )
        val_loss, val_acc, frame_auc, video_auc, n_vids = evaluate(
            model, val_loader, criterion, device
        )

        # Step scheduler (OneCycleLR steps per batch, CosineAnnealingLR per epoch)
        if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["frame_auc"].append(frame_auc)
        history["video_auc"].append(video_auc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}]  "
            f"Loss: {train_loss:.4f}/{val_loss:.4f}  "
            f"Acc: {train_acc:.3f}/{val_acc:.3f}  "
            f"AUC frame: {frame_auc:.4f}  video: {video_auc:.4f}  "
            f"({n_vids} vids)  LR: {current_lr:.2e}  ({elapsed:.1f}s)"
        )

        # Save if best video-level AUC (the realistic metric)
        if video_auc > best_auc:
            best_auc = video_auc
            save_checkpoint(
                model, save_path, args.model_name,
                val_auc=video_auc, val_acc=val_acc, img_size=args.img_size,
            )

    print(f"\n🏆 Best Video-Level AUC: {best_auc:.4f}")
    print(f"   Checkpoint: {save_path}")

    os.makedirs(args.model_dir, exist_ok=True)
    plot_history(history, args.model_dir)

    print("\n💡 To use your model with the API:")
    print(f"   MODEL_PATH={save_path} uvicorn main:app --port 8000")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B4 on video frames for deepfake detection."
    )
    p.add_argument("--data_dir",    default="./video_frames",
        help="Frame dataset dir with train/ and val/ (from extract_video_frames.py)")
    p.add_argument("--model_dir",   default="./models",
        help="Where to save checkpoints and training curves")
    p.add_argument("--model_name",  default="efficientnet_b4",
        help="timm model name (default: efficientnet_b4)")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=2e-4,
        help="Peak learning rate (OneCycleLR)")
    p.add_argument("--dropout",     type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--tc_loss_weight", type=float, default=0.1,
        help="Temporal consistency loss weight (0 to disable, default 0.1)")
    p.add_argument("--pos_weight",  type=float, default=1.0,
        help="BCEWithLogitsLoss pos_weight — increase if fake frames are rare")
    p.add_argument("--freeze_epochs", type=int, default=2,
        help="Freeze backbone for this many epochs at start (warm-up head first)")
    p.add_argument("--balance_sampler", action="store_true",
        help="Use WeightedRandomSampler to balance real/fake frames per batch")
    args = p.parse_args()
    main(args)
