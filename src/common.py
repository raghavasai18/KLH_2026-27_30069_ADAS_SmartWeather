
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_weather_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    classes = checkpoint["class_names"]

    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, classes, device


def predict_weather(model, classes, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

    return (
        classes[predicted.item()],
        confidence.item(),
        probabilities[0].cpu().tolist()
    )
