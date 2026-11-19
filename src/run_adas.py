
import argparse
from pathlib import Path
import json
import cv2
from ultralytics import YOLO
from src.common import load_weather_model, predict_weather


def make_alerts(weather, detections):
    alerts = []

    if weather.lower() in ["rain", "fog", "snow", "sand"]:
        alerts.append(
            f"Adverse weather predicted: {weather}. Drive cautiously."
        )

    classes = {d["class"].lower() for d in detections}

    if "person" in classes:
        alerts.append("Pedestrian detected: remain attentive.")

    if "motorcycle" in classes:
        alerts.append("Motorcycle detected: maintain awareness.")

    if classes.intersection({"car", "bus", "truck", "bicycle"}):
        alerts.append("Vehicles detected: maintain a safe distance.")

    if not alerts:
        alerts.append("No rule-based alert triggered.")

    return alerts


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--weather-model",
        default="models/dawn_weather_resnet18.pth"
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise SystemExit("Image not found: " + str(image_path))

    # Weather classification
    weather_model, classes, device = load_weather_model(
        args.weather_model
    )

    weather, weather_conf, weather_scores = predict_weather(
        weather_model, classes, device, image_path
    )

    # YOLO object detection
    detector = YOLO(args.yolo_model)

    results = detector.predict(
        source=str(image_path),
        conf=args.conf,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls.item())

        detections.append({
            "class": detector.names[cls_id],
            "confidence": round(float(box.conf.item()), 4),
            "xyxy": [
                round(float(v), 2)
                for v in box.xyxy[0].tolist()
            ]
        })

    alerts = make_alerts(weather, detections)

    # Annotate output image
    annotated = results[0].plot()

    banner = f"Weather: {weather} ({weather_conf:.1%})"
    cv2.rectangle(
        annotated, (0, 0),
        (annotated.shape[1], 42),
        (20, 20, 20), -1
    )

    cv2.putText(
        annotated, banner,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_out = output_dir / f"{image_path.stem}_adas.jpg"
    json_out = output_dir / f"{image_path.stem}_results.json"

    cv2.imwrite(str(image_out), annotated)

    report = {
        "weather_prediction": weather,
        "weather_confidence": weather_conf,
        "weather_scores": weather_scores,
        "detections": detections,
        "alerts": alerts,
        "limitations": [
            "YOLO uses pretrained weights.",
            "Alerts are informational only.",
            "No collision probability or braking distance is calculated."
        ]
    }

    with open(json_out, "w") as f:
        json.dump(report, f, indent=4)

    print("\nSMART WEATHER ADAS RESULTS")
    print("Weather:", weather)
    print("Confidence:", round(weather_conf * 100, 2), "%")
    print("Objects detected:", len(detections))

    for d in detections:
        print(d["class"], d["confidence"])

    print("\nAlerts:")
    for alert in alerts:
        print("-", alert)

    print("\nAnnotated image:", image_out)
    print("JSON report:", json_out)


if __name__ == "__main__":
    main()
