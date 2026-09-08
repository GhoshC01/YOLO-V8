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
OCR_ENGINES = ("paddle", "easy", "rapid")

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# Standard: MH12AB1234. Also series-less UT plates: LA020749 / LA 02 0749.
PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}\d{1,2}(?:[A-Z]{1,3}\d{1,4}|\d{3,4})$"
)
# Two-line commercial / auto plates: "AP13" above "V7951"
PLATE_LINE_TOP = re.compile(r"^[A-Z]{2}\d{1,2}$")
PLATE_LINE_BOTTOM = re.compile(r"^[A-Z]{0,3}\d{1,4}$")
MIN_READ_SIDE = 480

CROP_SIDE_PAD = 0.12
CROP_RIGHT_EXTRA = 0.18
MIN_CROP_PAD_PX = 10
MIN_CROP_HEIGHT = 40
MIN_CROP_WIDTH = 120
CROP_BORDER_RATIO = 0.06
DETECT_IMGSZ_SET = (640, 1280, 1600)
JUNK_OCR = re.compile(
    r"(GPS|GITUDE|NGITUDE|LONGIT|LATITUD|CHANNEL|TIMESTAMP|HTTP|WWW|INDIA)",
    re.I,
)
DIGIT_HEAVY = re.compile(r"^\d{6,}$")

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
    if engine not in OCR_ENGINES:
        raise ValueError(f"Unknown OCR engine: {engine}. Choose from {OCR_ENGINES}.")

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
        elif engine == "rapid":
            from rapidocr import RapidOCR

            ocr = RapidOCR()
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
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    # Camera plates often OCR the IND hologram strip as a prefix.
    if cleaned.startswith("IND") and len(cleaned) > 8:
        cleaned = cleaned[3:]
    return cleaned


def normalize_plate_candidate(text):
    """Clean OCR text and drop trailing bumper/hologram junk if needed."""
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""
    if PLATE_PATTERN.match(cleaned):
        return cleaned
    # Extra trailing char(s) from IND strip / reflector (e.g. LA020749C).
    for trim in range(1, 3):
        if len(cleaned) > trim and PLATE_PATTERN.match(cleaned[:-trim]):
            return cleaned[:-trim]
    return cleaned


def merge_plate_line_texts(texts):
    """Combine OCR lines from two-line Indian plates into one string.

    Auto / commercial plates often read as separate lines (AP13 + V7951).
    Prefer a joined candidate when it matches the plate pattern.
    """
    cleaned = []
    for text in texts:
        value = normalize_plate_candidate(str(text))
        if value and not JUNK_OCR.search(value):
            cleaned.append(value)
    if not cleaned:
        return ""

    candidates = list(cleaned)
    candidates.append(normalize_plate_candidate("".join(cleaned)))
    for i, top in enumerate(cleaned):
        for j, bottom in enumerate(cleaned):
            if i == j:
                continue
            if PLATE_LINE_TOP.match(top) and PLATE_LINE_BOTTOM.match(bottom):
                candidates.append(normalize_plate_candidate(top + bottom))

    return max(candidates, key=score_plate_text)


def score_plate_text(text):
    """Prefer typical Indian plate length / pattern."""
    if not text:
        return -1
    if JUNK_OCR.search(text) or DIGIT_HEAVY.match(text):
        return -5
    # GPS leftovers like UDE783320130000E / 1D7370200
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.7 and not PLATE_PATTERN.match(text):
        return -3
    if text.endswith(("E", "N", "W", "S")) and digit_ratio > 0.5 and not PLATE_PATTERN.match(text):
        return -4

    score = len(text)
    if 7 <= len(text) <= 11:
        score += 5
    if 8 <= len(text) <= 10:
        score += 3
    if PLATE_PATTERN.match(text):
        score += 20
    # State-code start is a strong Indian-plate signal.
    if re.match(r"^[A-Z]{2}\d", text):
        score += 8
    return score


def ensure_min_read_size(img):
    """Upscale tiny uploads so YOLO/OCR can resolve two-line plates."""
    if img is None or img.size == 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max(MIN_READ_SIDE / max(h, 1), MIN_READ_SIDE / max(w, 1), 1.0)
    if scale <= 1.05:
        return img, 1.0
    resized = cv2.resize(
        img,
        (max(int(w * scale), 1), max(int(h * scale), 1)),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized, scale


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


def deskew_plate(crop):
    """Rotate a skewed plate crop upright (handles angled CCTV views)."""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(thr) > 127:
        thr = 255 - thr
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 40:
        return crop
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 1.5 or abs(angle) > 40:
        return crop
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        crop,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

def pad_box(
    x1,
    y1,
    x2,
    y2,
    img_h,
    img_w,
    side_pad=CROP_SIDE_PAD,
    right_extra=CROP_RIGHT_EXTRA,
):
    """Expand a YOLO box before OCR so edge characters are not cut off.

    Close-up plates already fill most of the frame — use smaller absolute padding
    so bumper lights / GPS text are not pulled into the OCR crop.
    """
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    # Large boxes (close-ups): keep padding tiny. Small distant plates: keep % pad.
    if bw >= 280:
        left = top = bottom = 6
        right = 14
    else:
        left = max(int(bw * side_pad), MIN_CROP_PAD_PX)
        top = max(int(bh * side_pad), MIN_CROP_PAD_PX)
        bottom = max(int(bh * side_pad), MIN_CROP_PAD_PX)
        right = max(int(bw * (side_pad + right_extra)), MIN_CROP_PAD_PX + 2)
    return (
        max(0, x1 - left),
        max(0, y1 - top),
        min(img_w, x2 + right),
        min(img_h, y2 + bottom),
    )


def extract_plate_crop(img, x1, y1, x2, y2, right_extra=CROP_RIGHT_EXTRA):
    """Build an OCR-ready plate crop with padding, upscaling, and a soft border."""
    img_h, img_w = img.shape[:2]
    px1, py1, px2, py2 = pad_box(
        x1, y1, x2, y2, img_h, img_w, right_extra=right_extra
    )
    crop = img[py1:py2, px1:px2]
    if crop.size == 0:
        return None, {"x1": px1, "y1": py1, "x2": px2, "y2": py2}

    crop = crop.copy()
    h, w = crop.shape[:2]
    scale = max(MIN_CROP_HEIGHT / h, MIN_CROP_WIDTH / w, 1.0)
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (max(int(w * scale), 1), max(int(h * scale), 1)),
            interpolation=cv2.INTER_CUBIC,
        )

    border = max(4, int(min(crop.shape[:2]) * CROP_BORDER_RATIO))
    crop = cv2.copyMakeBorder(
        crop,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=(210, 210, 210),
    )
    return crop, {"x1": px1, "y1": py1, "x2": px2, "y2": py2}


def _box_aspect(x1, y1, x2, y2):
    return (x2 - x1) / max(y2 - y1, 1)


def _looks_like_osd_box(x1, y1, x2, y2, img_h, img_w):
    """Drop timestamp / GPS overlay boxes that YOLO often confuses as plates."""
    bw, bh = x2 - x1, y2 - y1
    if bw < 20 or bh < 10:
        return True
    # Timestamp strip near the top edge
    if y2 < img_h * 0.18 and _box_aspect(x1, y1, x2, y2) > 2.5:
        return True
    # Very bottom GPS caption band
    if y1 > img_h * 0.78 and bh < img_h * 0.25:
        return True
    # Tiny relative to frame
    if (bw * bh) < (img_w * img_h * 0.01):
        return True
    return False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


def collect_detection_boxes(detector, img, conf, imgsz):
    """Run YOLO at multiple sizes; keep overlaps so OCR can pick the true plate."""
    img_h, img_w = img.shape[:2]
    sizes = sorted({640, int(imgsz), 1280})
    raw = []
    for size in sizes:
        for result in detector(img, conf=min(conf, 0.12), imgsz=size, verbose=False):
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                if _looks_like_osd_box(x1, y1, x2, y2, img_h, img_w):
                    continue
                raw.append((float(box.conf), x1, y1, x2, y2))

    # Prefer plate-like geometry in the vehicle band (not bottom GPS strip).
    def box_priority(item):
        conf_i, x1, y1, x2, y2 = item
        ar = _box_aspect(x1, y1, x2, y2)
        cy = (y1 + y2) / 2.0 / img_h
        aspect_score = 1.0 if 0.7 <= ar <= 6.0 else 0.2
        band_score = 1.0 if 0.25 <= cy <= 0.85 else 0.3
        return (aspect_score + band_score, conf_i)

    raw.sort(key=box_priority, reverse=True)
    kept = []
    for cand in raw:
        conf_i, x1, y1, x2, y2 = cand
        # Soft NMS: only drop near-identical boxes so angled crops survive.
        if any(_iou((x1, y1, x2, y2), (k[1], k[2], k[3], k[4])) > 0.75 for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= 8:
            break
    return kept


def _mask_to_proposals(img, mask, score=0.34, max_boxes=3):
    img_h, img_w = img.shape[:2]
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 18:
            continue
        ar = w / max(h, 1)
        # Single-line plates are wide; two-line auto plates are nearer square.
        if ar < 0.7 or ar > 8.0:
            continue
        if (w * h) < (img_w * img_h * 0.008):
            continue
        if _looks_like_osd_box(x, y, x + w, y + h, img_h, img_w):
            continue
        proposals.append((score, x, y, x + w, y + h))
    proposals.sort(key=lambda b: (b[3] - b[1]) * (b[4] - b[2]), reverse=True)
    return proposals[:max_boxes]


def color_plate_proposals(img, max_boxes=4):
    """Color heuristics for yellow (day/night) and white private plates at any angle."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    proposals = []
    # Daytime yellow commercial plates
    proposals.extend(
        _mask_to_proposals(img, cv2.inRange(hsv, (15, 60, 60), (42, 255, 255)), 0.36)
    )
    # Low-light / night yellow (weaker saturation)
    proposals.extend(
        _mask_to_proposals(img, cv2.inRange(hsv, (12, 25, 40), (45, 255, 255)), 0.33)
    )
    # White / light private plates
    proposals.extend(
        _mask_to_proposals(img, cv2.inRange(hsv, (0, 0, 150), (180, 60, 255)), 0.32)
    )

    kept = []
    for cand in proposals:
        if any(_iou((cand[1], cand[2], cand[3], cand[4]), (k[1], k[2], k[3], k[4])) > 0.5 for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_boxes:
            break
    return kept


def yellow_plate_proposals(img, max_boxes=3):
    """Backward-compatible alias."""
    return color_plate_proposals(img, max_boxes=max_boxes)

def ocr_box_variants(img, read_fn, x1, y1, x2, y2, conf):
    """OCR a box with raw / padded / deskewed crops — handles angled CCTV plates."""
    candidates = []

    def consider(crop, crop_box):
        if crop is None or crop.size == 0:
            return
        text = best_ocr_text(read_fn, crop)
        if text and not JUNK_OCR.search(text) and score_plate_text(text) > 0:
            candidates.append((text, crop_box))

    raw = img[y1:y2, x1:x2]
    consider(raw, {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    consider(deskew_plate(raw), {"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    crop, crop_box = extract_plate_crop(img, x1, y1, x2, y2)
    consider(crop, crop_box)
    consider(deskew_plate(crop) if crop is not None else None, crop_box)

    if not any(PLATE_PATTERN.match(t) and len(t) >= 7 for t, _ in candidates):
        wide_crop, wide_box = extract_plate_crop(
            img, x1, y1, x2, y2, right_extra=CROP_RIGHT_EXTRA + 0.22
        )
        consider(wide_crop, wide_box)
        consider(deskew_plate(wide_crop) if wide_crop is not None else None, wide_box)

    if not candidates:
        return None

    text, crop_box = max(candidates, key=lambda item: score_plate_text(item[0]))
    return {
        "text": text,
        "confidence": round(float(conf), 4),
        "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "crop_box": crop_box,
        "valid_format": bool(PLATE_PATTERN.match(text)),
        "score": score_plate_text(text),
    }


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
    return merge_plate_line_texts(_extract_paddle_texts(result))


def _read_with_easy(reader, plate_img):
    ocr_result = reader.readtext(
        plate_img,
        allowlist=PLATE_CHARS,
        detail=1,
        paragraph=False,
    )
    return merge_plate_line_texts([res[1] for res in ocr_result])


def _read_with_rapid(ocr, plate_img):
    # Detection-first for two-line plates (AP13 / V7951). Recognition-only often
    # collapses both lines into one garbled string on small crops.
    texts = []
    det = ocr(plate_img, use_det=True, use_cls=False)
    if det is not None and det.txts:
        texts.extend(det.txts)

    merged = merge_plate_line_texts(texts)
    if PLATE_PATTERN.match(merged):
        return merged

    result = ocr(plate_img, use_det=False, use_cls=False)
    if result is not None and result.txts:
        texts.extend(result.txts)

    return merge_plate_line_texts(texts)


def make_reader(engine="paddle"):
    """Return a callable that turns a plate crop into a cleaned text string."""
    ocr = get_ocr(engine)
    readers = {
        "paddle": lambda crop: _read_with_paddle(ocr, crop),
        "easy": lambda crop: _read_with_easy(ocr, crop),
        "rapid": lambda crop: _read_with_rapid(ocr, crop),
    }
    return readers[engine]


def best_ocr_text(read_fn, cropped_plate):
    """Read several renderings of the same crop and keep the best plate string."""
    if cropped_plate is None or cropped_plate.size == 0:
        return ""
    candidates = [
        read_fn(cropped_plate),
        read_fn(enhance_plate(cropped_plate, scale=2)),
    ]
    return max(candidates, key=score_plate_text)


def scene_ocr_plates(img, engine="rapid"):
    """Find plates by reading the middle of the frame (skips timestamp / GPS OSD).

    Works well for angled CCTV cars where YOLO confuses GPS text with plates.
    Also merges stacked two-line plate fragments into one plate string.
    """
    h, w = img.shape[:2]
    y0, y1 = int(h * 0.12), int(h * 0.82)
    mid = img[y0:y1, :]
    ocr = get_ocr(engine if engine in OCR_ENGINES else "rapid")
    try:
        result = ocr(mid, use_det=True, use_cls=True)
    except Exception:
        return []

    if result is None or not result.txts:
        return []

    scores = list(result.scores) if result.scores else [0.55] * len(result.txts)
    boxes = list(result.boxes) if result.boxes is not None else [None] * len(result.txts)
    fragments = []
    for text, score, box in zip(result.txts, scores, boxes):
        cleaned = normalize_plate_candidate(str(text))
        if not cleaned or JUNK_OCR.search(cleaned):
            continue

        if box is not None:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            x1, x2 = int(max(0, min(xs))), int(min(w, max(xs)))
            yy1, yy2 = int(max(0, min(ys) + y0)), int(min(h, max(ys) + y0))
        else:
            x1, yy1, x2, yy2 = 0, y0, w, y1

        fragments.append(
            {
                "text": cleaned,
                "confidence": round(float(score), 4),
                "box": {"x1": x1, "y1": yy1, "x2": x2, "y2": yy2},
                "cy": (yy1 + yy2) / 2.0,
            }
        )

    found = []
    used = set()
    fragments.sort(key=lambda f: f["cy"])
    for i, top in enumerate(fragments):
        if i in used:
            continue
        merged = None
        for j, bottom in enumerate(fragments):
            if i == j or j in used:
                continue
            if bottom["cy"] <= top["cy"]:
                continue
            # Same vertical stack (two-line plate), not distant captions.
            if abs(
                ((top["box"]["x1"] + top["box"]["x2"]) / 2.0)
                - ((bottom["box"]["x1"] + bottom["box"]["x2"]) / 2.0)
            ) > max(w * 0.25, 40):
                continue
            if bottom["cy"] - top["cy"] > h * 0.35:
                continue
            candidate = merge_plate_line_texts([top["text"], bottom["text"]])
            if PLATE_PATTERN.match(candidate):
                merged = (candidate, top, bottom, j)
                break
        if merged:
            text, a, b, j = merged
            used.add(i)
            used.add(j)
            x1 = min(a["box"]["x1"], b["box"]["x1"])
            y1 = min(a["box"]["y1"], b["box"]["y1"])
            x2 = max(a["box"]["x2"], b["box"]["x2"])
            y2 = max(a["box"]["y2"], b["box"]["y2"])
            found.append(
                {
                    "text": text,
                    "confidence": round((a["confidence"] + b["confidence"]) / 2.0, 4),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "valid_format": True,
                    "score": score_plate_text(text),
                }
            )

    for index, frag in enumerate(fragments):
        if index in used:
            continue
        text_score = score_plate_text(frag["text"])
        if text_score < 15 and not PLATE_PATTERN.match(frag["text"]):
            continue
        found.append(
            {
                "text": frag["text"],
                "confidence": frag["confidence"],
                "box": frag["box"],
                "crop_box": frag["box"],
                "valid_format": bool(PLATE_PATTERN.match(frag["text"])),
                "score": text_score,
            }
        )

    found.sort(key=lambda p: (p["score"], p["confidence"]), reverse=True)
    return found


def read_plates(
    img,
    model_path=DEFAULT_MODEL,
    conf=DEFAULT_CONF,
    imgsz=DEFAULT_IMGSZ,
    engine="paddle",
    include_invalid=False,
):
    """Detect plates in a BGR image and OCR each one.

    Handles close-ups, night CCTV, and angled cars. Returns dicts with
    text, confidence, box, crop_box, valid_format.
    """
    detector = get_detector(model_path)
    read_fn = make_reader(engine)
    img, _ = ensure_min_read_size(img)

    plates = []
    seen_text = set()

    # 1) Scene OCR on mid-frame — strongest for angled / distant CCTV plates.
    for plate in scene_ocr_plates(img, engine=engine):
        if not plate["valid_format"] and not include_invalid:
            continue
        if plate["text"] in seen_text:
            continue
        seen_text.add(plate["text"])
        plates.append(plate)

    # Early return when scene OCR already found a solid Indian plate.
    if any(p["valid_format"] and len(p["text"]) >= 7 for p in plates):
        plates.sort(key=lambda p: (p["score"], p["confidence"]), reverse=True)
        for plate in plates:
            plate.pop("score", None)
        return plates

    # 2) Fallback: color proposals + YOLO boxes (close-ups / hard crops).
    boxes = color_plate_proposals(img)
    for box in collect_detection_boxes(detector, img, conf=conf, imgsz=imgsz):
        if any(_iou((box[1], box[2], box[3], box[4]), (b[1], b[2], b[3], b[4])) > 0.75 for b in boxes):
            continue
        boxes.append(box)

    for det_conf, x1, y1, x2, y2 in boxes[:8]:
        plate = ocr_box_variants(img, read_fn, x1, y1, x2, y2, det_conf)
        if plate is None:
            continue
        if not plate["valid_format"] and not include_invalid:
            continue
        if plate["text"] in seen_text:
            continue
        seen_text.add(plate["text"])
        plates.append(plate)

    plates.sort(key=lambda p: (p["score"], p["confidence"]), reverse=True)
    for plate in plates:
        plate.pop("score", None)
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


def _dms_to_decimal(deg, minutes, seconds, hemi=""):
    value = abs(float(deg)) + float(minutes) / 60.0 + float(seconds) / 3600.0
    if hemi.upper() in ("S", "W"):
        value = -value
    return value


def _parse_coord(text):
    """Parse camera OSD latitude/longitude into decimal degrees."""
    if not text:
        return None
    raw = text.upper().replace(" ", "")

    dms = re.search(
        r"(\d{1,3})[°º\s]+(\d{1,2})['′\s]+(\d{1,2}(?:\.\d+)?)[\"″]?\s*([NSEW])?",
        text,
        re.I,
    )
    if dms:
        hemi = dms.group(4) or ""
        if not hemi:
            if "LAT" in raw and "S" in raw:
                hemi = "S"
            elif "LON" in raw and "W" in raw:
                hemi = "W"
            elif "LAT" in raw:
                hemi = "N"
            elif "LON" in raw:
                hemi = "E"
        return _dms_to_decimal(dms.group(1), dms.group(2), dms.group(3), hemi)

    dec = re.search(r"(-?\d{1,3}\.\d+)\s*([NSEW])?", text)
    if dec:
        value = float(dec.group(1))
        hemi = (dec.group(2) or "").upper()
        if hemi in ("S", "W"):
            value = -abs(value)
        return value
    return None


def _parse_overlay_datetime(text):
    """Parse common CCTV timestamp styles into DD/MM/YYYY HH:MM:SS."""
    patterns = [
        (r"(\d{4})[-/](\d{2})[-/](\d{2})[ T](\d{2}):(\d{2}):(\d{2})", "ymd"),
        (r"(\d{2})[-/](\d{2})[-/](\d{4})[ T](\d{2}):(\d{2}):(\d{2})", "dmy"),
    ]
    for pattern, order in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        a, b, c, hh, mm, ss = match.groups()
        if order == "ymd":
            yyyy, mo, dd = a, b, c
        else:
            dd, mo, yyyy = a, b, c
        return f"{dd}/{mo}/{yyyy} {hh}:{mm}:{ss}"
    return None


def extract_camera_overlay(img, engine="rapid"):
    """Read GPS + timestamp from CCTV OSD bands (top / bottom of the frame)."""
    h, w = img.shape[:2]
    bands = [
        img[0 : max(int(h * 0.16), 40), :],
        img[max(h - int(h * 0.22), 0) : h, :],
    ]

    ocr = get_ocr(engine if engine in OCR_ENGINES else "rapid")
    lines = []
    for band in bands:
        try:
            result = ocr(band, use_det=True, use_cls=False)
        except Exception:
            continue
        if result is None or not result.txts:
            continue
        lines.extend(str(t) for t in result.txts if t)

    latitude = ""
    longitude = ""
    date_time = ""
    joined = " | ".join(lines)

    for line in lines:
        upper = line.upper()
        if "LAT" in upper and not latitude:
            value = _parse_coord(line)
            if value is not None:
                latitude = f"{value:.2f}"
        if ("LON" in upper or "LONG" in upper) and not longitude:
            value = _parse_coord(line)
            if value is not None:
                longitude = f"{value:.2f}"
        if not date_time:
            parsed = _parse_overlay_datetime(line)
            if parsed:
                date_time = parsed

    if not date_time:
        parsed = _parse_overlay_datetime(joined)
        if parsed:
            date_time = parsed

    return {
        "latitude": latitude,
        "longitude": longitude,
        "date_time": date_time,
        "overlay_text": lines,
    }


def format_client_payload(plates, overlay=None, now=None):
    """Client response shape: Plate_Number, lat/long, confidence%, date_time."""
    from datetime import datetime

    overlay = overlay or {}
    valid = [p for p in plates if p.get("valid_format")]
    best = valid[0] if valid else None
    plate_number = best["text"] if best else ""
    confidence = f"{best['confidence'] * 100:.2f}%" if best else "0.00%"
    date_time = overlay.get("date_time") or (now or datetime.now()).strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    return {
        "Plate_Number": plate_number,
        "latitude": overlay.get("latitude") or "",
        "longitude": overlay.get("longitude") or "",
        "confidence": confidence,
        "date_time": date_time,
    }