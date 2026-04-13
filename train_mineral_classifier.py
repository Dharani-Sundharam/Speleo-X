"""
=============================================================================
SPELEO-X | MINERAL CLASSIFIER TRAINING SCRIPT
=============================================================================
Dataset  : Minerals Identification Dataset (7 classes, ~5640 images)
           biotite | bornite | chrysocolla | malachite | muscovite |
           pyrite  | quartz
Model    : MobileNetV2 (ImageNet pretrained) + custom top layers
Output   : mineral_classifier.h5  +  class_indices.json
=============================================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset", "minet")
MODEL_OUT     = os.path.join(BASE_DIR, "mineral_classifier.h5")
IDX_OUT       = os.path.join(BASE_DIR, "class_indices.json")

IMG_SIZE      = (160, 160)   # MobileNetV2 minimum recommended
BATCH_SIZE    = 32
EPOCHS_FROZEN = 10           # Train only head (feature extractor frozen)
EPOCHS_FINETUNE = 10         # Unfreeze top 40 layers and fine-tune
VAL_SPLIT     = 0.2
SEED          = 42

print(f"\n{'='*62}")
print("  SPELEO-X  ·  MINERAL CLASSIFIER TRAINING")
print(f"{'='*62}")
print(f"  Dataset   : {DATASET_DIR}")
print(f"  Model out : {MODEL_OUT}")
print(f"  Classes   : {', '.join(sorted(os.listdir(DATASET_DIR)))}")
print(f"{'='*62}\n")

# ──────────────────────────────────────────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale           = 1.0 / 255,
    rotation_range    = 30,
    width_shift_range = 0.15,
    height_shift_range= 0.15,
    shear_range       = 0.15,
    zoom_range        = 0.20,
    horizontal_flip   = True,
    brightness_range  = [0.75, 1.25],
    validation_split  = VAL_SPLIT,
)
val_gen = ImageDataGenerator(
    rescale          = 1.0 / 255,
    validation_split = VAL_SPLIT,
)

train_ds = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "training",
    seed         = SEED,
    shuffle      = True,
)
val_ds = val_gen.flow_from_directory(
    DATASET_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = "categorical",
    subset       = "validation",
    seed         = SEED,
    shuffle      = False,
)

NUM_CLASSES = len(train_ds.class_indices)
print(f"[DATA]  Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")
print(f"[DATA]  Classes ({NUM_CLASSES}): {train_ds.class_indices}\n")

# Save class index mapping
with open(IDX_OUT, "w") as f:
    json.dump(train_ds.class_indices, f, indent=2)
print(f"[✓] class_indices.json saved → {IDX_OUT}\n")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL  —  MobileNetV2 + Custom Head
# ──────────────────────────────────────────────────────────────────────────────
base = MobileNetV2(
    input_shape = (*IMG_SIZE, 3),
    include_top = False,
    weights     = "imagenet",
)
base.trainable = False   # Phase 1: freeze base

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(base.input, out)

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

print(f"[MODEL] Parameters (trainable): "
      f"{model.count_params():,}  /  "
      f"total: {model.count_params():,}\n")

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Train Head (Base Frozen)
# ──────────────────────────────────────────────────────────────────────────────
print("─── Phase 1 : Head Training (base frozen) ─────────────────────")
cb_phase1 = [
    callbacks.EarlyStopping(patience=4, restore_best_weights=True,
                            monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, verbose=1),
]
model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FROZEN, callbacks=cb_phase1, verbose=1,
)

# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Fine-tune Top 40 Layers
# ──────────────────────────────────────────────────────────────────────────────
print("\n─── Phase 2 : Fine-tuning (top 40 base layers unfrozen) ────────")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer = optimizers.Adam(learning_rate=1e-4),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

cb_phase2 = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                            monitor="val_accuracy"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, verbose=1),
    callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True,
                              monitor="val_accuracy", verbose=1),
]
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FINETUNE, callbacks=cb_phase2, verbose=1,
)

# Final save (in case EarlyStopping didn't trigger checkpoint)
model.save(MODEL_OUT)
val_acc = max(history.history.get("val_accuracy", [0.0]))
print(f"\n[✓] Model saved → {MODEL_OUT}")
print(f"[✓] Best val accuracy : {val_acc*100:.2f}%\n")
print("Training complete. Run spectral_pipeline.py to use the model.")
