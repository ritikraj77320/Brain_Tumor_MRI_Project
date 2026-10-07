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

TEST_DIR = (
    PROJECT_ROOT
    / "image_dataset"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "efficientnetb0_best.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

PLOTS_DIR = (
    OUTPUT_DIR
    / "plots"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)

print(
    "        EFFICIENTNETB0 TEST EVALUATION"
)

print("=" * 70)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

CLASS_NAMES_PATH = (
    OUTPUT_DIR
    / "class_names.json"
)

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


print("\nClasses:")

for index, name in enumerate(class_names):

    print(
        f"{index}: {name}"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = (
    tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        labels="inferred",

        label_mode="int",

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        shuffle=False

    )
)


print(
    f"\nTest images: "
    f"{len(test_dataset.file_paths)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best EfficientNetB0 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating on unseen test data...")

test_loss, test_accuracy = model.evaluate(

    test_dataset,

    verbose=1

)


print("\n" + "=" * 70)

print("                 TEST RESULTS")

print("=" * 70)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []

y_pred = []

y_probability = []


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

    y_probability.extend(
        predictions
    )


y_true = np.array(y_true)

y_pred = np.array(y_pred)

y_probability = np.array(
    y_probability
)


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


print("\n" + "=" * 70)

print("                 TEST METRICS")

print("=" * 70)

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

print("\n" + "=" * 70)

print("              CLASSIFICATION REPORT")

print("=" * 70)


report = classification_report(

    y_true,

    y_pred,

    target_names=class_names,

    digits=4,

    zero_division=0

)

print(report)


report_path = (

    OUTPUT_DIR

    / "efficientnetb0_classification_report.txt"

)


with open(

    report_path,

    "w"

) as file:

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

    "EfficientNetB0 - Confusion Matrix"

)


plt.tight_layout()


cm_path = (

    PLOTS_DIR

    / "efficientnetb0_confusion_matrix.png"

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

    "f1_score": float(f1),

    "test_loss": float(test_loss)

}


metrics_path = (

    OUTPUT_DIR

    / "efficientnetb0_metrics.json"

)


with open(

    metrics_path,

    "w"

) as file:

    json.dump(

        metrics,

        file,

        indent=4

    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_path = (

    OUTPUT_DIR

    / "efficientnetb0_predictions.csv"

)


import pandas as pd


prediction_data = {

    "true_label": [

        class_names[i]

        for i in y_true

    ],

    "predicted_label": [

        class_names[i]

        for i in y_pred

    ],

    "confidence": [

        float(np.max(probability))

        for probability in y_probability

    ]

}


prediction_df = pd.DataFrame(

    prediction_data

)


prediction_df.to_csv(

    predictions_path,

    index=False

)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)

print(
    "       EFFICIENTNETB0 EVALUATION COMPLETED"
)

print("=" * 70)

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

print(
    f"✓ {predictions_path}"
)

print("\n" + "=" * 70)