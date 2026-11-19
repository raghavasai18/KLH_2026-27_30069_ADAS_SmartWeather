import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from common import build_resnet18, IMAGE_SIZE, MEAN, STD


def main():
    p = argparse.ArgumentParser(description="Train ResNet-18 weather classifier")
    p.add_argument("--data", required=True, help="ImageFolder root with train/ and val/ (or validation/)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output", default="models/dawn_weather_resnet18.pth")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()

    root = Path(args.data)
    train_dir = root / "train"
    val_dir = root / "val" if (root / "val").exists() else root / "validation"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit("Expected --data/train/<class> and --data/val/<class> (or validation/<class>). See README.")

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"Train/validation classes differ: {train_ds.classes} vs {val_ds.classes}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(len(train_ds.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = -1.0
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train(); running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * images.size(0)
        model.eval(); correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(1)
                correct += (predictions == labels).sum().item(); total += labels.size(0)
        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={running_loss/len(train_ds):.4f} | val_accuracy={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": train_ds.classes,
                        "image_size": IMAGE_SIZE, "mean": MEAN, "std": STD,
                        "validation_accuracy": best_acc}, output)
            print(f"Saved best checkpoint: {output}")
    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    print("Note: use a separate untouched test split for final paper results.")

if __name__ == "__main__":
    main()
