#!/usr/bin/env python3
"""
train.py — Fine-tune YOLOv8n on plant disease data (PlantDoc via Roboflow)
Keeps all 80 COCO object classes and ADDS 30 plant disease classes.

Usage:
    # 1. Get your free Roboflow API key at https://roboflow.com
    # 2. Run:
    python Agro_Bot/train.py --api-key YOUR_KEY

    # Or if you already downloaded the dataset manually:
    python Agro_Bot/train.py --dataset-dir Agro_Bot/dataset

Requirements:
    pip install roboflow ultralytics
"""

import argparse
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = "yolov8n.pt"          # COCO-pretrained starting point
OUTPUT_MODEL  = "Agro_Bot/plant_model.pt"
EPOCHS        = 50
IMG_SIZE      = 640
BATCH         = 16                    # Reduce to 8 if GPU OOM
DEVICE        = "cuda"
PROJECT_DIR   = "Agro_Bot/runs"


def download_dataset(api_key: str, dest: str) -> str:
    """Download PlantDoc dataset from Roboflow in YOLOv8 format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    print("[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    # PlantDoc: 30 plant disease classes, 2,569 images
    project = rf.workspace("joseph-nelson").project("plantdoc")
    dataset  = project.version(1).download("yolov8", location=dest)
    return dataset.location


def train(dataset_yaml: str, epochs: int = EPOCHS, batch: int = BATCH):
    """Fine-tune YOLOv8n on the downloaded dataset."""
    from ultralytics import YOLO

    print(f"[INFO] Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"[INFO] Starting training for {epochs} epochs on {DEVICE}...")
    results = model.train(
        data    = dataset_yaml,
        epochs  = epochs,
        imgsz   = IMG_SIZE,
        batch   = batch,
        device  = DEVICE,
        project = PROJECT_DIR,
        name    = "plant_detection",
        patience= 15,
        save    = True,
        plots   = True,
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,
        flipud  = 0.1,
        fliplr  = 0.5,
        mosaic  = 1.0,
    )

    best = Path(PROJECT_DIR) / "plant_detection" / "weights" / "best.pt"
    if best.exists():
        import shutil
        shutil.copy(best, OUTPUT_MODEL)
        print(f"\n[INFO] ✅ Training complete! Best model saved to: {OUTPUT_MODEL}")
        print(f"[INFO] Update laptop_receiver.py:  YOLO_MODEL = \"{OUTPUT_MODEL}\"")
    else:
        print(f"[WARN] Check {PROJECT_DIR}/plant_detection/weights/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on PlantDoc dataset")
    parser.add_argument("--api-key",     type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="Agro_Bot/dataset")
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--batch",       type=int, default=BATCH)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    yaml_path   = os.path.join(dataset_dir, "data.yaml")

    if not os.path.exists(yaml_path):
        if not args.api_key:
            print("[ERROR] Dataset not found. Provide --api-key to download from Roboflow.")
            print("        Get a free key at: https://app.roboflow.com/settings/api")
            sys.exit(1)
        dataset_dir = download_dataset(args.api_key, dataset_dir)
        yaml_path   = os.path.join(dataset_dir, "data.yaml")

    print(f"[INFO] Using dataset: {yaml_path}")
    train(yaml_path, epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
