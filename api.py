"""FastAPI service for uploading images and reading Indian number plates."""

import base64
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import detection_store
import plate_reader

ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "static" / "index.html"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

detection_store.ensure_storage_dir()

app = FastAPI(
    title="Number Plate Detector API",
    description="Upload any image format and get the detected number plates back.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_image(upload: UploadFile, data: bytes):
    if not data:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{upload.filename}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    img = plate_reader.decode_image(data)
    if img is None:
        raise HTTPException(
            status_code=415,
            detail=f"'{upload.filename}' is not a supported image format.",
        )
    return img


def _build_client_result(
    img,
    plates,
    ocr_engine,
    latitude="",
    longitude="",
    date_time="",
):
    overlay = plate_reader.extract_camera_overlay(img, engine=ocr_engine)
    if latitude:
        overlay["latitude"] = latitude
    if longitude:
        overlay["longitude"] = longitude
    if date_time:
        overlay["date_time"] = date_time
    return plate_reader.format_client_payload(
        plates, overlay=overlay, now=datetime.now()
    )


@app.on_event("startup")
def warm_up_models():
    """Load YOLO + OCR at boot so the first request is not slow."""
    detection_store.ensure_storage_dir()
    plate_reader.get_detector()
    plate_reader.make_reader("rapid")


app.mount(
    "/detections",
    StaticFiles(directory=str(detection_store.STORAGE_DIR)),
    name="detections",
)


@app.get("/", response_class=HTMLResponse)
def upload_page():
    if UI_FILE.exists():
        return HTMLResponse(UI_FILE.read_text())
    return HTMLResponse("<h1>Number Plate Detector API</h1><p>See /docs</p>")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": plate_reader.DEFAULT_MODEL,
        "storage_dir": str(detection_store.STORAGE_DIR),
        "ocr_engines": list(plate_reader.OCR_ENGINES),
    }


@app.get("/detections/list")
def detections_list(limit: int = 50):
    """List recent saved detections for client demos."""
    items = detection_store.list_detections(limit=limit)
    return {"count": len(items), "items": items}


@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(..., description="One or more images of any format"),
    conf: float = Form(plate_reader.DEFAULT_CONF),
    imgsz: int = Form(plate_reader.DEFAULT_IMGSZ),
    ocr: str = Form("rapid"),
    include_invalid: bool = Form(False),
    annotate: bool = Form(False),
    save: bool = Form(True),
    latitude: str = Form(""),
    longitude: str = Form(""),
    date_time: str = Form(""),
):
    """Detect plates and return client payload.

    Response shape (single image):
    {
      "Plate_Number": "MH46AF1865",
      "latitude": "21.13",
      "longitude": "79.70",
      "confidence": "58.23%",
      "date_time": "03/09/2026 13:08:03"
    }
    """
    if ocr not in plate_reader.OCR_ENGINES:
        raise HTTPException(
            status_code=400,
            detail=f"ocr must be one of: {', '.join(plate_reader.OCR_ENGINES)}.",
        )

    payloads = []
    for upload in files:
        data = await upload.read()
        img = _load_image(upload, data)

        plates = plate_reader.read_plates(
            img,
            conf=conf,
            imgsz=imgsz,
            engine=ocr,
            include_invalid=include_invalid,
        )
        payload = _build_client_result(
            img,
            plates,
            ocr_engine=ocr,
            latitude=latitude,
            longitude=longitude,
            date_time=date_time,
        )

        if save:
            detection_store.save_detection(img, plates, upload.filename)

        if annotate:
            annotated = plate_reader.annotate(img, plates)
            jpeg = plate_reader.encode_jpeg(annotated)
            payload["annotated_image"] = (
                "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            )

        payloads.append(payload)

    # One image → exact client object. Multiple → list of those objects.
    return payloads[0] if len(payloads) == 1 else payloads


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(..., description="A single image of any format"),
    conf: float = Form(plate_reader.DEFAULT_CONF),
    imgsz: int = Form(plate_reader.DEFAULT_IMGSZ),
    ocr: str = Form("rapid"),
    include_invalid: bool = Form(False),
    save: bool = Form(True),
):
    """Same as /detect but responds with the annotated JPEG itself."""
    data = await file.read()
    img = _load_image(file, data)

    plates = plate_reader.read_plates(
        img,
        conf=conf,
        imgsz=imgsz,
        engine=ocr,
        include_invalid=include_invalid,
    )
    annotated = plate_reader.annotate(img, plates)
    headers = {"X-Plates": ",".join(p["text"] for p in plates) or "none"}

    if save:
        saved = detection_store.save_detection(img, plates, file.filename)
        headers["X-Saved-Url"] = saved["annotated_url"]
        headers["X-Saved-Id"] = saved["id"]

    return Response(
        content=plate_reader.encode_jpeg(annotated),
        media_type="image/jpeg",
        headers=headers,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=False)
