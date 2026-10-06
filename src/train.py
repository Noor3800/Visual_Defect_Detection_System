import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import create_model


SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

DATA_ROOT = Path("data")
MODEL_ROOT = Path("models")
RESULTS_ROOT = Path("results")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_ROOT / "val", transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_dataset, train_loader, val_loader


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

    return total_loss / total


def main():
    set_seed()

    MODEL_ROOT.mkdir(exist_ok=True)
    RESULTS_ROOT.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, train_loader, val_loader = get_loaders()

    print("Classes:", train_dataset.classes)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_loader.dataset))

    model = create_model(num_classes=len(train_dataset.classes)).to(device)

    # Calculate inverse-frequency class weights from the training set.
    counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=len(train_dataset.classes),
    ).float()

    weights = counts.sum() / (len(counts) * counts)
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")

    history = []

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_ROOT / "best_model.pth",
            )

            with open(MODEL_ROOT / "class_names.json", "w", encoding="utf-8") as f:
                json.dump(train_dataset.classes, f, indent=2)

    with open(RESULTS_ROOT / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nBest model saved to models/best_model.pth")


if __name__ == "__main__":
    main()
