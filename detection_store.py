"""Save detection outputs to disk for client demos."""

import json
import re
import time
from pathlib import Path

import cv2

import plate_reader

ROOT = Path(__file__).resolve().parent
STORAGE_DIR = ROOT / "tmp_detections"


def ensure_storage_dir():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return STORAGE_DIR


def _safe_name(name, limit=40):
    stem = Path(name or "image").stem
    cleaned = re.sub(r"[^\w.-]+", "_", stem).strip("._")
    return (cleaned or "image")[:limit]


def save_detection(img, plates, source_name="upload"):
    """Write annotated image, plate crops, and JSON metadata under tmp_detections/."""
    root = ensure_storage_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{stamp}_{_safe_name(source_name)}"
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=False)

    annotated_path = folder / "annotated.jpg"
    cv2.imwrite(str(annotated_path), plate_reader.annotate(img, plates))

    saved_plates = []
    for index, plate in enumerate(plates, start=1):
        box = plate["box"]
        crop, _ = plate_reader.extract_plate_crop(
            img, box["x1"], box["y1"], box["x2"], box["y2"]
        )
        crop_name = f"plate_{index}_{plate['text'] or 'unknown'}.jpg"
        crop_path = folder / crop_name
        if crop is not None and crop.size:
            cv2.imwrite(str(crop_path), crop)

        saved_plates.append(
            {
                **plate,
                "crop_file": crop_name if crop is not None and crop.size else None,
                "crop_url": f"/detections/{folder_name}/{crop_name}"
                if crop is not None and crop.size
                else None,
            }
        )

    meta = {
        "id": folder_name,
        "source": source_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plate_count": len(plates),
        "best_plate": plates[0]["text"] if plates else None,
        "plates": saved_plates,
        "annotated_file": "annotated.jpg",
        "annotated_url": f"/detections/{folder_name}/annotated.jpg",
        "folder": str(folder.relative_to(root)),
    }
    (folder / "result.json").write_text(json.dumps(meta, indent=2))

    return meta


def list_detections(limit=50):
    """Return recent saved detections, newest first."""
    root = ensure_storage_dir()
    entries = []

    for folder in sorted(root.iterdir(), reverse=True):
        if not folder.is_dir():
            continue

        meta_file = folder / "result.json"
        if meta_file.exists():
            try:
                entries.append(json.loads(meta_file.read_text()))
            except json.JSONDecodeError:
                continue
        else:
            entries.append(
                {
                    "id": folder.name,
                    "folder": folder.name,
                    "annotated_url": f"/detections/{folder.name}/annotated.jpg",
                }
            )

        if len(entries) >= limit:
            break

    return entries
