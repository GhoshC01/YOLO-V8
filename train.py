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

        # --- Advanced Augmentation for Diverse Fonts & Styles ---
        degrees=15,       # rotation of the font
        perspective=0.001, # different angles of the license plate
        shear=10,         # italic font style
        scale=0.5,        # size of the font
        hsv_h=0.015,      # multiple colors of the license plate (yellow, green, white)
        hsv_s=0.7,
        hsv_v=0.4
        
    )


if __name__ == "__main__":
    main()
