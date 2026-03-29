"""
FastAPI wrapper for Lipi OCR inference.

Provides HTTP endpoints for single-image and batch recognition.

Usage:
    uvicorn deploy.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import io
from typing import Optional

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError("FastAPI required: pip install fastapi uvicorn python-multipart")

from deploy.server import LipiServer


# Initialize server (loaded once at startup)
app = FastAPI(
    title="Lipi OCR API",
    description="Multilingual OCR for Indian scripts",
    version="1.0.0",
)

server: Optional[LipiServer] = None


class RecognitionResult(BaseModel):
    text: str
    confidence: float
    script_id: str


class BatchResult(BaseModel):
    results: list[RecognitionResult]


def preprocess_upload(image_bytes: bytes, target_height: int = 32) -> np.ndarray:
    """Convert uploaded image bytes to preprocessed array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    new_w = int(w * target_height / h)
    new_w = min(max(new_w, 1), 320)
    img = img.resize((new_w, target_height), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)  # CHW


@app.on_event("startup")
async def startup():
    global server
    import os
    model_dir = os.environ.get("LIPI_MODEL_DIR", "./models")
    server = LipiServer(model_dir)


@app.post("/recognize", response_model=RecognitionResult)
async def recognize(image: UploadFile = File(...)):
    """Recognize text in a single word crop image."""
    if server is None:
        raise HTTPException(503, "Server not initialized")

    image_bytes = await image.read()
    crop = preprocess_upload(image_bytes)
    results = server.process_page([crop])

    if not results:
        raise HTTPException(500, "Recognition failed")

    r = results[0]
    return RecognitionResult(
        text=r.text, confidence=r.confidence, script_id=r.script_id
    )


@app.post("/recognize/batch", response_model=BatchResult)
async def recognize_batch(images: list[UploadFile] = File(...)):
    """Recognize text in a batch of word crop images."""
    if server is None:
        raise HTTPException(503, "Server not initialized")

    crops = []
    for img_file in images:
        image_bytes = await img_file.read()
        crops.append(preprocess_upload(image_bytes))

    results = server.process_page(crops)

    return BatchResult(
        results=[
            RecognitionResult(
                text=r.text, confidence=r.confidence, script_id=r.script_id
            )
            for r in results
        ]
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "languages_loaded": list(server.packs.keys()) if server else [],
    }
