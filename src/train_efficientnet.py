from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "image_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

PHASE1_EPOCHS = 15

PHASE2_EPOCHS = 15

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("             EFFICIENTNETB0 TRANSFER LEARNING")
print("=" * 70)

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

print("\nLoading validation dataset...")

valid_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 4. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="data_augmentation"
)


# ============================================================
# 6. PERFORMANCE
# ============================================================

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

class_counts = {
    "glioma": 564,
    "meningioma": 358,
    "no_tumor": 335,
    "pituitary": 438
}

labels = []

for class_name, count in class_counts.items():

    class_index = class_names.index(class_name)

    labels.extend(
        [class_index] * count
    )

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        np.unique(labels),
        weights
    )
}

print("\nClass weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index]:<15} : {weight:.4f}"
    )


# ============================================================
# 8. BUILD EFFICIENTNETB0
# ============================================================

print("\nBuilding EfficientNetB0...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Phase 1: freeze pretrained network

base_model.trainable = False


# ============================================================
# 9. CLASSIFICATION HEAD
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

# EfficientNetB0 includes its own input preprocessing
x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Dropout(
    0.4
)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(
    0.3
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = tf.keras.Model(
    inputs,
    outputs,
    name="Brain_Tumor_EfficientNetB0"
)


# ============================================================
# 10. PHASE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\nModel created successfully.")

print(
    f"Trainable parameters: "
    f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
)


# ============================================================
# 11. PHASE 1 CALLBACKS
# ============================================================

best_model_path = (
    MODELS_DIR
    / "efficientnetb0_best.keras"
)

callbacks_phase1 = [

    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# 12. PHASE 1 TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1: FEATURE EXTRACTION")
print("=" * 70)

history1 = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=PHASE1_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase1
)


# ============================================================
# 13. PHASE 2: FINE-TUNING
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING")
print("=" * 70)

print("\nUnfreezing upper EfficientNet layers...")


base_model.trainable = True


# Freeze the earlier layers
# Fine-tune only the upper portion

fine_tune_from = 180

for layer in base_model.layers[:fine_tune_from]:

    layer.trainable = False


# Keep BatchNormalization frozen
# to make fine-tuning more stable

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# 14. RECOMPILE WITH SMALL LEARNING RATE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


trainable_count = sum(
    tf.size(v).numpy()
    for v in model.trainable_variables
)


print(
    f"\nTrainable parameters after fine-tuning: "
    f"{trainable_count:,}"
)


# ============================================================
# 15. PHASE 2 CALLBACKS
# ============================================================

callbacks_phase2 = [

    tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# 16. PHASE 2 TRAINING
# ============================================================

history2 = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=PHASE2_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks_phase2
)


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODELS_DIR
    / "efficientnetb0_final.keras"
)

model.save(
    final_model_path
)

print(
    f"\nFinal model saved to:\n"
    f"{final_model_path}"
)


# ============================================================
# 18. COMBINE HISTORY
# ============================================================

history = {}

for key in history1.history:

    history[key] = (
        history1.history[key]
        +
        history2.history.get(key, [])
    )


# ============================================================
# 19. SAVE HISTORY
# ============================================================

history_path = (
    OUTPUTS_DIR
    / "efficientnetb0_history.json"
)

with open(history_path, "w") as file:

    json.dump(

        {
            key: [
                float(value)
                for value in values
            ]

            for key, values in history.items()
        },

        file,

        indent=4
    )


# ============================================================
# 20. TRAINING PLOTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "EfficientNetB0 - Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    PLOTS_DIR
    / "efficientnetb0_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(10, 6))

plt.plot(
    history["loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "EfficientNetB0 - Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    PLOTS_DIR
    / "efficientnetb0_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. BEST RESULT
# ============================================================

best_val_accuracy = max(
    history["val_accuracy"]
)

best_val_loss = min(
    history["val_loss"]
)


print("\n" + "=" * 70)
print("       EFFICIENTNETB0 TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.4f}"
)

print("\nGenerated:")

print(
    f"✓ {best_model_path}"
)

print(
    f"✓ {final_model_path}"
)

print(
    f"✓ {history_path}"
)

print(
    "✓ EfficientNetB0 accuracy plot"
)

print(
    "✓ EfficientNetB0 loss plot"
)

print("\n" + "=" * 70)