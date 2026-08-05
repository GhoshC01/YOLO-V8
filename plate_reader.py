"""Shared number plate detection + OCR core used by both the CLI and the API."""

import io
import re
import threading

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "runs/detect/my_plate_model-2/weights/best.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 1280

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")

_lock = threading.Lock()
_detectors = {}
_ocr_engines = {}


def get_detector(model_path=DEFAULT_MODEL):
    """Load YOLO weights once per process."""
    with _lock:
        if model_path not in _detectors:
            _detectors[model_path] = YOLO(model_path)
        return _detectors[model_path]


def get_ocr(engine="paddle"):
    """Load the OCR engine once per process."""
    with _lock:
        if engine in _ocr_engines:
            return _ocr_engines[engine]

        if engine == "paddle":
            from paddleocr import PaddleOCR

            try:
                ocr = PaddleOCR(lang="en", use_textline_orientation=True)
            except TypeError:
                ocr = PaddleOCR(lang="en")
        elif engine == "easy":
            import easyocr

            ocr = easyocr.Reader(["en"], gpu=False)
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")

        _ocr_engines[engine] = ocr
        return ocr


def decode_image(data):
    """Decode image bytes from any common format into a BGR numpy array.

    Handles JPEG/PNG/WebP/BMP/TIFF/GIF via Pillow, HEIC/HEIF from iPhones via
    pillow-heif, and falls back to OpenCV for anything Pillow rejects.
    """
    try:
        from PIL import Image, ImageOps

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass

        with Image.open(io.BytesIO(data)) as pil_img:
            # Honour EXIF rotation so phone photos are upright
            pil_img = ImageOps.exif_transpose(pil_img)
            rgb = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception:
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)


def clean_plate_text(text):
    """Keep only A-Z / 0-9 and return an uppercase plate string."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def score_plate_text(text):
    """Prefer typical Indian plate length / pattern."""
    if not text:
        return -1
    score = len(text)
    if 8 <= len(text) <= 11:
        score += 5
    if PLATE_PATTERN.match(text):
        score += 10
    return score


def enhance_plate(cropped_plate, scale=2):
    """Contrast-boost an upscaled copy; crop bounds stay unchanged."""
    h, w = cropped_plate.shape[:2]
    upscaled = cv2.resize(
        cropped_plate,
        (max(int(w * scale), 1), max(int(h * scale), 1)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)


def _extract_paddle_texts(result):
    if not result:
        return []
    first = result[0]
    if isinstance(first, dict):
        return first.get("rec_texts") or []
    rec_texts = getattr(first, "rec_texts", None)
    if rec_texts:
        return list(rec_texts)
    if isinstance(first, list):
        texts = []
        for line in first:
            if line and len(line) >= 2:
                item = line[1]
                texts.append(item[0] if isinstance(item, (list, tuple)) else str(item))
        return texts
    return []


def _read_with_paddle(ocr, plate_img):
    result = ocr.predict(plate_img) if hasattr(ocr, "predict") else ocr.ocr(plate_img)
    return clean_plate_text("".join(_extract_paddle_texts(result)))


def _read_with_easy(reader, plate_img):
    ocr_result = reader.readtext(
        plate_img,
        allowlist=PLATE_CHARS,
        detail=1,
        paragraph=False,
    )
    return clean_plate_text("".join(res[1] for res in ocr_result))


def make_reader(engine="paddle"):
    """Return a callable that turns a plate crop into a cleaned text string."""
    ocr = get_ocr(engine)
    if engine == "paddle":
        return lambda crop: _read_with_paddle(ocr, crop)
    return lambda crop: _read_with_easy(ocr, crop)


def best_ocr_text(read_fn, cropped_plate):
    """Read several renderings of the same crop and keep the best plate string."""
    candidates = [
        read_fn(cropped_plate),
        read_fn(enhance_plate(cropped_plate, scale=2)),
        read_fn(enhance_plate(cropped_plate, scale=4)),
    ]
    return max(candidates, key=score_plate_text)


def read_plates(
    img,
    model_path=DEFAULT_MODEL,
    conf=DEFAULT_CONF,
    imgsz=DEFAULT_IMGSZ,
    engine="paddle",
    include_invalid=False,
):
    """Detect plates in a BGR image and OCR each one.

    Returns a list of dicts: text, confidence, box, valid_format.
    Boxes whose text does not match the plate pattern are dropped unless
    include_invalid is True.
    """
    detector = get_detector(model_path)
    read_fn = make_reader(engine)

    results = detector(img, conf=conf, imgsz=imgsz, verbose=False)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            text = best_ocr_text(read_fn, crop)
            valid = bool(PLATE_PATTERN.match(text))
            if not valid and not include_invalid:
                continue

            plates.append(
                {
                    "text": text,
                    "confidence": round(float(box.conf), 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "valid_format": valid,
                }
            )

    plates.sort(key=lambda p: p["confidence"], reverse=True)
    return plates


def annotate(img, plates):
    """Draw green boxes and plate text on a copy of the image."""
    canvas = img.copy()
    for plate in plates:
        box = plate["box"]
        cv2.rectangle(
            canvas, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2
        )
        if plate["text"]:
            cv2.putText(
                canvas,
                plate["text"],
                (box["x1"], max(box["y1"] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
    return canvas


def encode_jpeg(img, quality=90):
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    return buffer.tobytes()
