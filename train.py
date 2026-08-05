"""Train a YOLOv8 model for Indian license plate detection."""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"


def main():
    # Ensure data.yaml uses an absolute dataset root (Ultralytics resolves relative
    # path: . against the project CWD, not the yaml folder).
    yaml_text = f"""path: {ROOT / 'dataset'}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['indian_licence_plate']
"""
    DATA_YAML.write_text(yaml_text)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=16,
        name="my_plate_model",
    )


if __name__ == "__main__":
    main()
