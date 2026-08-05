"""FastAPI service for uploading images and reading Indian number plates."""

import base64
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response

import plate_reader

ROOT = Path(__file__).resolve().parent
UI_FILE = ROOT / "static" / "index.html"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

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


@app.on_event("startup")
def warm_up_models():
    """Load YOLO + OCR at boot so the first request is not slow."""
    plate_reader.get_detector()
    plate_reader.make_reader("paddle")


@app.get("/", response_class=HTMLResponse)
def upload_page():
    if UI_FILE.exists():
        return HTMLResponse(UI_FILE.read_text())
    return HTMLResponse("<h1>Number Plate Detector API</h1><p>See /docs</p>")


@app.get("/health")
def health():
    return {"status": "ok", "model": plate_reader.DEFAULT_MODEL}


@app.post("/detect")
async def detect(
    files: List[UploadFile] = File(..., description="One or more images of any format"),
    conf: float = Form(plate_reader.DEFAULT_CONF),
    imgsz: int = Form(plate_reader.DEFAULT_IMGSZ),
    ocr: str = Form("paddle"),
    include_invalid: bool = Form(False),
    annotate: bool = Form(False),
):
    """Detect and read number plates in the uploaded images.

    Set annotate=true to also receive the boxed image as a base64 JPEG.
    """
    if ocr not in ("paddle", "easy"):
        raise HTTPException(status_code=400, detail="ocr must be 'paddle' or 'easy'.")

    results = []
    for upload in files:
        data = await upload.read()
        img = _load_image(upload, data)

        started = time.perf_counter()
        plates = plate_reader.read_plates(
            img,
            conf=conf,
            imgsz=imgsz,
            engine=ocr,
            include_invalid=include_invalid,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        entry = {
            "filename": upload.filename,
            "width": img.shape[1],
            "height": img.shape[0],
            "plate_count": len(plates),
            "plates": plates,
            "best_plate": plates[0]["text"] if plates else None,
            "processing_ms": elapsed_ms,
        }

        if annotate:
            annotated = plate_reader.annotate(img, plates)
            jpeg = plate_reader.encode_jpeg(annotated)
            entry["annotated_image"] = (
                "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            )

        results.append(entry)

    return {"count": len(results), "results": results}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(..., description="A single image of any format"),
    conf: float = Form(plate_reader.DEFAULT_CONF),
    imgsz: int = Form(plate_reader.DEFAULT_IMGSZ),
    ocr: str = Form("paddle"),
    include_invalid: bool = Form(False),
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

    return Response(
        content=plate_reader.encode_jpeg(annotated),
        media_type="image/jpeg",
        headers={"X-Plates": ",".join(p["text"] for p in plates) or "none"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
