from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "image_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# 2. CLASS INFORMATION
# ============================================================

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# 3. CHECK DATASET PATH
# ============================================================

print("\n" + "=" * 60)
print("          BRAIN TUMOR MRI DATASET ANALYSIS")
print("=" * 60)

print("\nProject Directory:")
print(PROJECT_ROOT)

print("\nDataset Directory:")
print(DATASET_DIR)


if not DATASET_DIR.exists():

    print("\n❌ ERROR: Dataset directory not found!")

    print("Expected location:")
    print(DATASET_DIR)

    raise SystemExit


print("\n✅ Dataset directory found.")


# ============================================================
# 4. COUNT IMAGES
# ============================================================

def count_images(directory):

    counts = {}

    for class_name in CLASSES:

        class_dir = directory / class_name

        if not class_dir.exists():

            counts[class_name] = 0
            continue

        image_files = [
            file
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        counts[class_name] = len(image_files)

    return counts


train_counts = count_images(TRAIN_DIR)

valid_counts = count_images(VALID_DIR)

test_counts = count_images(TEST_DIR)


# ============================================================
# 5. DISPLAY DATASET COUNTS
# ============================================================

print("\n" + "-" * 60)
print("TRAINING DATA")
print("-" * 60)

for class_name, count in train_counts.items():

    print(f"{class_name:<15}: {count}")


print("\n" + "-" * 60)
print("VALIDATION DATA")
print("-" * 60)

for class_name, count in valid_counts.items():

    print(f"{class_name:<15}: {count}")


print("\n" + "-" * 60)
print("TEST DATA")
print("-" * 60)

for class_name, count in test_counts.items():

    print(f"{class_name:<15}: {count}")


# ============================================================
# 6. TOTAL IMAGES
# ============================================================

total_train = sum(train_counts.values())

total_valid = sum(valid_counts.values())

total_test = sum(test_counts.values())

total_images = (
    total_train
    + total_valid
    + total_test
)


print("\n" + "-" * 60)
print("TOTAL DATASET")
print("-" * 60)

print(f"Training images   : {total_train}")

print(f"Validation images : {total_valid}")

print(f"Test images       : {total_test}")

print(f"Total images      : {total_images}")


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 60)
print("TRAINING CLASS DISTRIBUTION")
print("-" * 60)

train_df = pd.DataFrame({
    "Class": list(train_counts.keys()),
    "Images": list(train_counts.values())
})

print(train_df.to_string(index=False))


# ============================================================
# 8. CLASS IMBALANCE CHECK
# ============================================================

print("\n" + "-" * 60)
print("CLASS BALANCE ANALYSIS")
print("-" * 60)

max_count = max(train_counts.values())

min_count = min(train_counts.values())

if min_count == 0:

    print("⚠️ At least one class contains zero images.")

else:

    imbalance_ratio = max_count / min_count

    print(f"Maximum class size : {max_count}")

    print(f"Minimum class size : {min_count}")

    print(f"Imbalance ratio    : {imbalance_ratio:.2f}")

    if imbalance_ratio <= 1.5:

        print("✅ Dataset is relatively balanced.")

    elif imbalance_ratio <= 2.0:

        print("⚠️ Moderate class imbalance detected.")

    else:

        print("⚠️ Significant class imbalance detected.")


# ============================================================
# 9. IMAGE DIMENSION ANALYSIS
# ============================================================

print("\n" + "-" * 60)
print("IMAGE DIMENSION ANALYSIS")
print("-" * 60)


def analyze_dimensions(directory):

    dimensions = []

    checked_images = 0

    for class_name in CLASSES:

        class_dir = directory / class_name

        if not class_dir.exists():
            continue

        for image_file in class_dir.iterdir():

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:

                with Image.open(image_file) as image:

                    dimensions.append(image.size)

                    checked_images += 1

            except Exception:

                pass

    return dimensions, checked_images


dimensions, checked_images = analyze_dimensions(TRAIN_DIR)


if dimensions:

    dimension_df = pd.DataFrame(
        dimensions,
        columns=["Width", "Height"]
    )

    print(f"Images checked: {checked_images}")

    print("\nMost common image dimensions:")

    print(
        dimension_df
        .value_counts()
        .head(10)
    )

    print("\nMinimum dimensions:")

    print(
        dimension_df.min()
    )

    print("\nMaximum dimensions:")

    print(
        dimension_df.max()
    )


# ============================================================
# 10. CORRUPTED IMAGE CHECK
# ============================================================

print("\n" + "-" * 60)
print("CORRUPTED IMAGE CHECK")
print("-" * 60)


def check_corrupted_images(directory):

    corrupted = []

    for class_name in CLASSES:

        class_dir = directory / class_name

        if not class_dir.exists():
            continue

        for image_file in class_dir.iterdir():

            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:

                with Image.open(image_file) as image:

                    image.verify()

            except Exception:

                corrupted.append(image_file)

    return corrupted


corrupted_train = check_corrupted_images(TRAIN_DIR)

corrupted_valid = check_corrupted_images(VALID_DIR)

corrupted_test = check_corrupted_images(TEST_DIR)


print(
    f"Corrupted training images   : {len(corrupted_train)}"
)

print(
    f"Corrupted validation images : {len(corrupted_valid)}"
)

print(
    f"Corrupted test images       : {len(corrupted_test)}"
)


# ============================================================
# 11. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("                 ANALYSIS COMPLETED")
print("=" * 60)

print("\nDataset classes:")

for class_name in CLASSES:

    print(f"  ✓ {class_name}")

print("\nNext step:")

print(
    "Preprocessing → Augmentation → Custom CNN"
)

print("=" * 60)