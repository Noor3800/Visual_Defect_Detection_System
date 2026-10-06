"""
Prepare one MVTec AD category for the SparkAI binary classification task.

MVTec AD's official protocol trains on defect-free images and tests on
defective/defect-free images. This script creates a CUSTOM binary
classification dataset by combining:
    - train/good
    - test/good
    - test/<defect_type>/*
from one selected category.

This is intentionally NOT the official MVTec benchmark protocol.
"""

from pathlib import Path
import random
import shutil

from sklearn.model_selection import train_test_split


SEED = 42
CATEGORY = "bottle"

MVTEC_ROOT = Path("data/raw/mvtec_ad")
OUTPUT_ROOT = Path("data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(category: str):
    category_root = MVTEC_ROOT / category
    train_good = category_root / "train" / "good"
    test_root = category_root / "test"

    if not train_good.exists():
        raise FileNotFoundError(
            f"Could not find {train_good}. "
            "Place the extracted MVTec AD dataset under data/raw/mvtec_ad."
        )

    normal = [
        p for p in train_good.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # MVTec test/good images are additional normal examples.
    test_good = test_root / "good"
    if test_good.exists():
        normal.extend(
            p for p in test_good.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    defective = []
    for defect_dir in test_root.iterdir():
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue

        defective.extend(
            p for p in defect_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    return normal, defective


def split_and_copy(paths, class_name, seed=SEED):
    random.seed(seed)
    paths = sorted(paths)
    random.shuffle(paths)

    train, temp = train_test_split(
        paths,
        test_size=0.30,
        random_state=seed,
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
    )

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    for split_name, split_paths in splits.items():
        destination = OUTPUT_ROOT / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for index, source in enumerate(split_paths):
            # Include the original filename and parent directory to reduce
            # collisions between different defect types.
            safe_name = f"{source.parent.name}_{source.name}"
            target = destination / f"{index:05d}_{safe_name}"
            shutil.copy2(source, target)

    return {name: len(items) for name, items in splits.items()}


def main():
    normal, defective = collect_images(CATEGORY)

    print(f"MVTec category: {CATEGORY}")
    print(f"Normal images found: {len(normal)}")
    print(f"Defective images found: {len(defective)}")

    for split in ["train", "val", "test"]:
        for cls in ["normal", "defective"]:
            folder = OUTPUT_ROOT / split / cls
            if folder.exists():
                shutil.rmtree(folder)

    normal_counts = split_and_copy(normal, "normal")
    defective_counts = split_and_copy(defective, "defective")

    print("\nCustom split:")
    print("Normal:    ", normal_counts)
    print("Defective: ", defective_counts)
    print("\nDone.")


if __name__ == "__main__":
    main()
