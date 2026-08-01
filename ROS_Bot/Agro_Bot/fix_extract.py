#!/usr/bin/env python3
"""
fix_extract.py — Extract PlantDoc zip using Windows extended path (\\?\)
This bypasses the 260-character MAX_PATH limit without requiring admin rights.

Usage:
    python Agro_Bot/fix_extract.py
"""
import zipfile, os, sys, shutil
from pathlib import Path

ZIP_PATH = r"C:\pd\roboflow.zip"
DEST     = r"C:\pd"

if not os.path.exists(ZIP_PATH):
    print(f"[ERROR] Zip not found: {ZIP_PATH}")
    sys.exit(1)

print(f"[INFO] Re-extracting {ZIP_PATH} with long-path support...")
print(f"[INFO] Destination: {DEST}\n")

ok, skipped = 0, 0
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    total = len(zf.namelist())
    for i, member in enumerate(zf.namelist(), 1):
        # Use \\?\ prefix for the target path to bypass MAX_PATH
        raw_target = os.path.join(DEST, member.replace('/', os.sep))
        target = "\\\\?\\" + raw_target  # extended-length path prefix

        if member.endswith('/'):
            os.makedirs(raw_target, exist_ok=True)
            continue

        # Make sure parent directory exists (using normal path — dirs are short)
        parent = os.path.dirname(raw_target)
        os.makedirs(parent, exist_ok=True)

        try:
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            ok += 1
        except Exception as e:
            skipped += 1

        if i % 200 == 0 or i == total:
            print(f"  {i}/{total} — OK:{ok}  Skipped:{skipped}", end='\r')

print(f"\n[INFO] Done! Extracted {ok} files, skipped {skipped}.")

# Verify key folders
for folder in ['train/images', 'valid/images', 'test/images']:
    path = os.path.join(DEST, folder)
    count = len(os.listdir(path)) if os.path.exists(path) else 0
    print(f"  {folder}: {count} files")

print("\n[INFO] Now run training:")
print("  python .\\Agro_Bot\\train.py --dataset-dir C:\\pd")
