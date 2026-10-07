from pathlib import Path
import json

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "image_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("Loading validation dataset...")

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("Loading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_names_path = OUTPUT_DIR / "class_names.json"

with open(class_names_path, "w") as file:
    json.dump(class_names, file, indent=4)

print(f"\nClass names saved to: {class_names_path}")


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.10
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# NORMALIZATION
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(
    1.0 / 255.0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

# Dataset counts obtained from your analysis
class_counts = {
    "glioma": 564,
    "meningioma": 358,
    "no_tumor": 335,
    "pituitary": 438
}


class_indices = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


y = []

for class_name, count in class_counts.items():

    class_index = class_indices[class_name]

    y.extend([class_index] * count)


y = np.array(y)


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)


class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        np.unique(y),
        weights
    )
}


print("\nClass weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index]:<15} : {weight:.4f}"
    )


# ============================================================
# DATASET PERFORMANCE
# ============================================================

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("              PREPROCESSING READY")

print("=" * 60)

print(f"\nImage size : {IMAGE_SIZE}")

print(f"Batch size : {BATCH_SIZE}")

print(f"Classes    : {len(class_names)}")

print("\nAugmentation:")

print("✓ Horizontal flip")
print("✓ Rotation")
print("✓ Zoom")
print("✓ Translation")
print("✓ Contrast adjustment")

print("\nClass weighting:")

print("✓ Enabled")

print("\nDataset split:")

print("✓ Training")
print("✓ Validation")
print("✓ Test")

print("\n" + "=" * 60)