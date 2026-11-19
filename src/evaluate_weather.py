import argparse
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from common import load_weather_model, IMAGE_SIZE, MEAN, STD
from torchvision import transforms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Test folder with one subfolder per class")
    p.add_argument("--checkpoint", default="models/dawn_weather_resnet18.pth")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, _ = load_weather_model(args.checkpoint, device)
    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(args.data, transform=tf)
    if ds.classes != classes:
        raise SystemExit(f"Test classes {ds.classes} do not match checkpoint classes {classes}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            y_true.extend(y.tolist()); y_pred.extend(pred.tolist())
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
