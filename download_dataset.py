"""
scripts/download_dataset.py
Downloads the '140k Real and Fake Faces' dataset from Kaggle and organises it.

Requirements:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API credentials.
    https://www.kaggle.com/account → Create New API Token

Usage:
    python scripts/download_dataset.py
"""

import os
import shutil
import zipfile
import random
from pathlib import Path
from tqdm import tqdm


DATASET_SLUG  = "xhlulu/140k-real-and-fake-faces"
DOWNLOAD_DIR  = "./downloads"
DATA_DIR      = "./data"
VAL_SPLIT     = 0.15   # 15% of images go to validation
RANDOM_SEED   = 42


def download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print("⬇️  Downloading dataset from Kaggle …")
    os.system(f"kaggle datasets download -d {DATASET_SLUG} -p {DOWNLOAD_DIR} --unzip")
    print("✅  Download complete.")


def reorganise():
    """
    Kaggle dataset structure:
        real_vs_fake/
            real-vs-fake/
                train/real/    train/fake/
                valid/real/    valid/fake/
                test/real/     test/fake/

    We copy into:
        data/
            train/real/   train/fake/
            val/real/     val/fake/
    """
    src_root = Path(DOWNLOAD_DIR) / "real_vs_fake" / "real-vs-fake"

    for split, dest_split in [("train", "train"), ("valid", "val"), ("test", "val")]:
        for cls in ["real", "fake"]:
            src  = src_root / split / cls
            dest = Path(DATA_DIR) / dest_split / cls
            if not src.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            files = list(src.glob("*"))
            print(f"  Copying {len(files)} {cls} images ({split} → {dest_split}) …")
            for f in tqdm(files, desc=f"  {split}/{cls}", leave=False):
                shutil.copy2(f, dest / f.name)

    print(f"\n📁 Dataset ready at {DATA_DIR}/")
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            p = Path(DATA_DIR) / split / cls
            if p.exists():
                print(f"  {split}/{cls}: {len(list(p.iterdir()))} images")


if __name__ == "__main__":
    download()
    reorganise()
    print("\n🎉 Done! You can now run:  python src/train.py --data_dir ./data")
