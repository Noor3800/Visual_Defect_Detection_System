import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import create_model


DATA_ROOT = Path("data")
MODEL_ROOT = Path("models")
RESULTS_ROOT = Path("results")


def main():
    RESULTS_ROOT.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    test_dataset = datasets.ImageFolder(
        DATA_ROOT / "test",
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(num_classes=len(test_dataset.classes))
    model.load_state_dict(
        torch.load(
            MODEL_ROOT / "best_model.pth",
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    with open(
        RESULTS_ROOT / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=test_dataset.classes,
    )
    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "confusion_matrix.png", dpi=200)
    plt.close()

    print(classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
