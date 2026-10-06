from pathlib import Path
from collections import Counter
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def analyze_dataset(data_dir: str = "data/raw"):
    root = Path(data_dir)
    class_counts = Counter()
    image_sizes = Counter()
    unreadable = []

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_dirs = [p for p in root.iterdir() if p.is_dir()]

    for class_dir in sorted(class_dirs):
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            class_counts[class_dir.name] += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1
            except Exception:
                unreadable.append(str(image_path))

    print("Class distribution:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")

    print(f"\nTotal images: {sum(class_counts.values())}")
    print("\nMost common image sizes:")
    for size, count in image_sizes.most_common(10):
        print(f"  {size}: {count}")

    print(f"\nUnreadable images: {len(unreadable)}")
    for path in unreadable:
        print(f"  {path}")

    return class_counts, image_sizes, unreadable


if __name__ == "__main__":
    analyze_dataset()
