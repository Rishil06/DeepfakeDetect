"""
train.py — Deepfake Detection Model Training
Uses EfficientNet-B4 via timm with fine-tuning for binary classification (real vs fake).
Dataset: 140k Real and Fake Faces (Kaggle) or any folder with real/ and fake/ subfolders.

Usage:
    python src/train.py --data_dir ./data --epochs 10 --batch_size 32
"""

import os
import argparse
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Expects directory structure:
        data_dir/
            real/   ← real images
            fake/   ← fake/AI-generated images
    """
    def __init__(self, data_dir: str, transform=None):
        self.samples = []
        self.transform = transform

        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Expected folder: {folder_path}")
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(folder_path, fname), label))

        print(f"  Loaded {len(self.samples)} images from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_model(model_name: str = "efficientnet_b4", pretrained: bool = True, dropout: float = 0.4):
    """
    Load a pretrained EfficientNet-B4 from timm and replace the classifier head
    with a binary output (real=0, fake=1).
    """
    model = timm.create_model(model_name, pretrained=pretrained, drop_rate=dropout)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1)
    )
    return model


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += imgs.size(0)
        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    auc = roc_auc_score(all_labels, all_probs)
    return running_loss / total, correct / total, auc


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")

    # Datasets
    train_tf, val_tf = get_transforms(img_size=args.img_size)
    train_dataset = DeepfakeDataset(os.path.join(args.data_dir, "train"), train_tf)
    val_dataset   = DeepfakeDataset(os.path.join(args.data_dir, "val"),   val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.model_name, pretrained=True, dropout=args.dropout)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc   = 0.0
    best_model = None
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"\n🚀 Training {args.model_name} for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}  "
              f"({elapsed:.1f}s)")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model = copy.deepcopy(model.state_dict())
            os.makedirs(args.model_dir, exist_ok=True)
            save_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save({
                "model_name": args.model_name,
                "state_dict": best_model,
                "val_auc": best_auc,
                "val_acc": val_acc,
                "img_size": args.img_size,
            }, save_path)
            print(f"  ✅ New best model saved (AUC={best_auc:.4f}) → {save_path}")

    print(f"\n🏆 Best Validation AUC: {best_auc:.4f}")

    # Plot training curves
    _plot_history(history, args.model_dir)


def _plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(epochs, history["val_auc"])
    axes[2].set_title("Val AUC")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=120)
    print(f"📊 Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument("--data_dir",    default="./data",   help="Root data dir with train/ and val/ subfolders")
    parser.add_argument("--model_dir",   default="./models", help="Where to save checkpoints")
    parser.add_argument("--model_name",  default="efficientnet_b4")
    parser.add_argument("--img_size",    type=int, default=224)
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--dropout",     type=float, default=0.4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
