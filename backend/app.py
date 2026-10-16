import io
import logging
import os
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eyesspies")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app = FastAPI(title="EyesSpies Vision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def load_models():
    from transformers import CLIPModel, CLIPProcessor
    from ultralytics import YOLO

    logger.info("Loading object detector: %s", YOLO_MODEL)
    detector = YOLO(YOLO_MODEL)
    logger.info("Loading scene model: %s", CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    scene_model = CLIPModel.from_pretrained(CLIP_MODEL)
    scene_model.eval()
    return detector, processor, scene_model


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": load_models.cache_info().currsize > 0}


@app.post("/api/analyze")
async def analyze(request: Request):
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG, PNG, or WebP image frame in the request body.",
        )

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image frame is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The request is not a valid image.") from exc

    try:
        detector, processor, scene_model = load_models()
        detections = detector.predict(image, verbose=False)[0]
        objects = [
            {
                "label": detections.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
            }
            for box in detections.boxes
        ]

        labels = [
            "an indoor scene",
            "an outdoor scene",
            "an urban scene",
            "a rural scene",
        ]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.inference_mode():
            scores = scene_model(**inputs).logits_per_image[0]
            setting_scores = torch.softmax(scores[:2], dim=0)
            environment_scores = torch.softmax(scores[2:], dim=0)

        setting_index = int(torch.argmax(setting_scores))
        environment_index = int(torch.argmax(environment_scores))
        return {
            "detections": objects,
            "context": {
                "setting": ["Indoor", "Outdoor"][setting_index],
                "setting_confidence": round(float(setting_scores[setting_index]), 4),
                "environment": ["Urban", "Rural"][environment_index],
                "environment_confidence": round(float(environment_scores[environment_index]), 4),
            },
        }
    except Exception as exc:
        logger.exception("Frame analysis failed")
        raise HTTPException(
            status_code=503,
            detail=f"Could not load or run the vision models: {exc}",
        ) from exc
