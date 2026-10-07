from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. PROJECT PATHS
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

EPOCHS = 30

SEED = 42


# ============================================================
# 3. LOAD DATASETS
# ============================================================

print("\n" + "=" * 60)
print("             CUSTOM CNN TRAINING")
print("=" * 60)

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
# 4. CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

NUM_CLASSES = len(class_names)

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# Save class names

class_names_path = OUTPUTS_DIR / "class_names.json"

with open(class_names_path, "w") as file:
    json.dump(class_names, file, indent=4)

print(f"\nClass names saved to:")
print(class_names_path)


# ============================================================
# 5. DATA AUGMENTATION
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
        )

    ],
    name="data_augmentation"
)


# ============================================================
# 6. PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

valid_dataset = valid_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

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
# 8. BUILD CUSTOM CNN
# ============================================================

print("\nBuilding Custom CNN...")


model = tf.keras.Sequential(

    [

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),


        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        data_augmentation,


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        tf.keras.layers.Rescaling(
            1.0 / 255.0
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 1
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 2
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 3
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 4
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),


        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING
        # ----------------------------------------------------

        tf.keras.layers.GlobalAveragePooling2D(),


        # ----------------------------------------------------
        # FULLY CONNECTED LAYER
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(
            0.5
        ),


        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ],

    name="Brain_Tumor_Custom_CNN"
)


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

print("\nModel Architecture:\n")

model.summary()


# ============================================================
# 10. COMPILE MODEL
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


# ============================================================
# 11. CALLBACKS
# ============================================================

best_model_path = (
    MODELS_DIR / "custom_cnn_best.keras"
)


callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=6,

        restore_best_weights=True,

        verbose=1
    ),


    tf.keras.callbacks.ModelCheckpoint(

        filepath=best_model_path,

        monitor="val_loss",

        save_best_only=True,

        verbose=1
    ),


    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6,

        verbose=1
    )

]


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)

print("                 STARTING TRAINING")

print("=" * 60)

print(f"\nEpochs     : {EPOCHS}")

print(f"Batch size : {BATCH_SIZE}")

print(f"Image size : {IMAGE_SIZE}")

print("\nTraining...\n")


history = model.fit(

    train_dataset,

    validation_data=valid_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks

)


# ============================================================
# 13. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODELS_DIR / "custom_cnn_final.keras"
)


model.save(final_model_path)


print("\nFinal model saved to:")

print(final_model_path)


# ============================================================
# 14. SAVE TRAINING HISTORY
# ============================================================

history_data = history.history

history_path = (
    OUTPUTS_DIR / "custom_cnn_history.json"
)


with open(history_path, "w") as file:

    json.dump(

        {
            key: [float(value) for value in values]

            for key, values
            in history_data.items()
        },

        file,

        indent=4

    )


print("\nTraining history saved to:")

print(history_path)


# ============================================================
# 15. TRAINING CURVES
# ============================================================

print("\nGenerating training plots...")


# Accuracy plot

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Custom CNN - Training vs Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

accuracy_plot = (
    PLOTS_DIR / "custom_cnn_accuracy.png"
)

plt.savefig(
    accuracy_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Loss plot

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Custom CNN - Training vs Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

loss_plot = (
    PLOTS_DIR / "custom_cnn_loss.png"
)

plt.savefig(
    loss_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. FINAL RESULT
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_val_loss = min(
    history.history["val_loss"]
)


print("\n" + "=" * 60)

print("              TRAINING COMPLETED")

print("=" * 60)

print(
    f"\nBest Validation Accuracy : "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best Validation Loss     : "
    f"{best_val_loss:.4f}"
)

print("\nGenerated files:")

print(
    f"✓ {best_model_path}"
)

print(
    f"✓ {final_model_path}"
)

print(
    f"✓ {accuracy_plot}"
)

print(
    f"✓ {loss_plot}"
)

print(
    f"✓ {history_path}"
)

print("\n" + "=" * 60)