import io
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.context_model import (
    CONTEXT_TASKS,
    ContextClassifier,
    context_from_probabilities,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eyesspies")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
CONTEXT_MODEL_PATH = Path(
    os.getenv(
        "CONTEXT_MODEL_PATH",
        str(Path(__file__).resolve().parent / "models" / "context_head.pt"),
    )
)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
inference_lock = threading.Lock()
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
        logger.info("Preloading vision models")
        await run_in_threadpool(load_models)
    yield


app = FastAPI(title="EyesSpies Vision API", version="0.1.0", lifespan=lifespan)
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


@lru_cache(maxsize=1)
def load_caption_model():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    logger.info("Loading image caption model: %s", BLIP_MODEL)
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def load_context_classifier():
    if not CONTEXT_MODEL_PATH.is_file():
        return None, ()

    checkpoint = torch.load(CONTEXT_MODEL_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("model_id") != CLIP_MODEL:
        raise ValueError(
            "The context checkpoint was trained with a different CLIP model. "
            "Set CLIP_MODEL to the training model or retrain the context head."
        )
    tasks = checkpoint.get("tasks", {})
    if not tasks or not set(tasks).issubset({"setting", "environment"}):
        raise ValueError("Context checkpoint must contain a supported trained task.")

    classifier = ContextClassifier(int(checkpoint["embedding_size"]))
    trained_tasks = tuple(tasks)
    incompatibility = classifier.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    expected_missing = {
        f"{head}.{parameter}"
        for head in {"setting", "environment"} - set(trained_tasks)
        for parameter in ("weight", "bias")
    }
    if set(incompatibility.missing_keys) != expected_missing or incompatibility.unexpected_keys:
        raise ValueError("Context checkpoint contains incomplete or unexpected classifier weights.")
    classifier.eval()
    for task, metadata in tasks.items():
        if tuple(metadata.get("labels", ())) != CONTEXT_TASKS[task]:
            raise ValueError(f"Context checkpoint labels do not match task '{task}'.")
        logger.info(
            "Loaded fine-tuned %s classifier (validation accuracy: %.3f)",
            task,
            float(metadata["validation_accuracy"]),
        )
    return classifier, trained_tasks


@app.get("/api/health")
def health():
    models_loaded = load_models.cache_info().currsize > 0
    return {
        "status": "ok" if models_loaded else "starting",
        "models_loaded": models_loaded,
        "fine_tuned_context_tasks": list(load_context_classifier()[1]),
    }


@app.post("/api/analyze")
async def analyze(
    request: Request,
    include_caption: bool = False,
    confidence_threshold: float = Query(default=0.35, ge=0.0, le=1.0),
):
    content_type = request.headers.get("content-type", "").split(";")[0].strip()
    if content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Send a JPEG, PNG, or WebP image frame in the request body.",
        )

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413, detail="The image frame exceeds the 5 MB limit."
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc

    image_bytes = bytearray()
    async for chunk in request.stream():
        image_bytes.extend(chunk)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="The image frame exceeds the 5 MB limit.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image frame is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as opened_image:
            if opened_image.width * opened_image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="The image dimensions are too large.")
            image = opened_image.convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=413, detail="The image dimensions are too large.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The request is not a valid image.") from exc

    if not inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="The vision model is busy. Please retry shortly.")

    try:
        return await run_in_threadpool(
            analyze_image, image, include_caption, confidence_threshold
        )
    except Exception as exc:
        logger.exception("Frame analysis failed")
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=503, detail="Vision analysis is temporarily unavailable.") from exc
    finally:
        inference_lock.release()


def analyze_image(
    image: Image.Image,
    include_caption: bool = False,
    confidence_threshold: float = 0.35,
):
    detector, processor, scene_model = load_models()
    context_classifier, trained_tasks = load_context_classifier()
    detections = detector.predict(
        image, verbose=False, conf=confidence_threshold
    )[0]
    objects = [
        {
            "label": detections.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
        }
        for box in detections.boxes
    ]

    setting_scores = None
    environment_scores = None
    needs_zero_shot = any(
        task not in trained_tasks for task in ("setting", "environment")
    )
    if needs_zero_shot:
        labels = [
            "an indoor scene",
            "an outdoor scene",
            "an urban scene",
            "a rural scene",
        ]
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        with torch.inference_mode():
            scores = scene_model(**inputs).logits_per_image[0]
            if "setting" not in trained_tasks:
                setting_scores = torch.softmax(scores[:2], dim=0)
            if "environment" not in trained_tasks:
                environment_scores = torch.softmax(scores[2:], dim=0)

    if trained_tasks:
        image_inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            embeddings = scene_model.get_image_features(
                pixel_values=image_inputs["pixel_values"]
            )
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            if "setting" in trained_tasks:
                setting_scores = torch.softmax(
                    context_classifier.setting(embeddings)[0], dim=0
                )
            if "environment" in trained_tasks:
                environment_scores = torch.softmax(
                    context_classifier.environment(embeddings)[0], dim=0
                )

    if setting_scores is None or environment_scores is None:
        raise RuntimeError("Scene context scores could not be produced.")
    context = context_from_probabilities(setting_scores, environment_scores)

    result = {
        "detections": objects,
        "context": context,
    }
    if include_caption:
        caption_processor, caption_model = load_caption_model()
        caption_inputs = caption_processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            caption_tokens = caption_model.generate(
                **caption_inputs,
                max_new_tokens=40,
                num_beams=3,
            )
        result["caption"] = caption_processor.decode(
            caption_tokens[0], skip_special_tokens=True
        ).strip()
    return result


frontend_dist = Path(__file__).resolve().parent.parent / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
