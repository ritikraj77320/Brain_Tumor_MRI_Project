from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "image_dataset" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "custom_cnn_best.keras"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

PLOTS_DIR = OUTPUT_DIR / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32


# ============================================================
# LOAD CLASS NAMES
# ============================================================

CLASS_NAMES_PATH = (
    OUTPUT_DIR / "class_names.json"
)

with open(CLASS_NAMES_PATH, "r") as file:

    class_names = json.load(file)


print("\n" + "=" * 60)

print("           CUSTOM CNN TEST EVALUATION")

print("=" * 60)


print("\nClasses:")

for index, name in enumerate(class_names):

    print(f"{index}: {name}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False

)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best Custom CNN...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully.")


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating model on test data...")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\nTest Loss     :", round(test_loss, 4))

print(
    "Test Accuracy :",
    round(test_accuracy, 4)
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []

y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)

print("                 TEST METRICS")

print("=" * 60)

print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)

print("              CLASSIFICATION REPORT")

print("=" * 60)

report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    digits=4,

    zero_division=0

)

print(report)


# Save report

report_path = (
    OUTPUT_DIR
    / "custom_cnn_classification_report.txt"
)

with open(report_path, "w") as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nConfusion Matrix:")

print(cm)


plt.figure(
    figsize=(8, 8)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names

)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Custom CNN - Confusion Matrix"
)

plt.tight_layout()


cm_path = (
    PLOTS_DIR
    / "custom_cnn_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "accuracy": float(accuracy),

    "precision": float(precision),

    "recall": float(recall),

    "f1_score": float(f1)

}


metrics_path = (
    OUTPUT_DIR
    / "custom_cnn_metrics.json"
)


with open(metrics_path, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print("             EVALUATION COMPLETED")

print("=" * 60)

print("\nGenerated files:")

print(
    f"✓ {report_path}"
)

print(
    f"✓ {cm_path}"
)

print(
    f"✓ {metrics_path}"
)

print("\n" + "=" * 60)