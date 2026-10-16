---
title: EyesSpies
emoji: 👁️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# EyesSpies

A browser-first computer-vision prototype. The React app reads a camera frame about once per second and sends it to the Python API. The API returns object detections from YOLO and indoor/outdoor plus urban/rural estimates from CLIP. When hosted, camera frames are sent to the server for inference and are not saved by this app.

## What you need

- Node.js 20 or newer
- Python 3.10 or newer
- A webcam (or phone camera if you open the site over HTTPS)

## Start the Python API

Open PowerShell in the project folder:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
backend\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload
```

The API listens at `http://localhost:8000`. On the first analysis request, it downloads the YOLO and CLIP model weights; this can take a while and requires an internet connection. It runs on CPU by default.

## Start the website

Open a second PowerShell window in the project folder:

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`) and allow camera access. Keep the API terminal running. Vite forwards `/api` requests to the API.

You can also use **Choose image** to test a JPEG, PNG, or WebP sample without granting camera access. Images must be smaller than 5 MB. The **Detection confidence** slider filters low-confidence object matches for both uploaded images and the next live frame. Still-image analysis includes an image-captioning model; its weights download on first use, so that first request may take longer. Live camera frames skip caption generation to keep the camera loop lighter.

## How it works

- `src/` contains the browser UI and camera capture code.
- `backend/app.py` serves `/api/health` and `/api/analyze`.
- The analyzer returns bounding boxes and confidence scores, plus one label for each of two separate context questions.
- Still-image analysis also generates a caption; live camera analysis skips captioning to reduce latency.
- Frames are processed in memory and are not written to disk.

The context scores are relative scores between the two labels in each category, not calibrated probabilities. This is a learning prototype: evaluate it on your own scenes and data before relying on its predictions.

## Train the scene-context classifier

The app starts with CLIP zero-shot predictions. A supervised training script can train separate indoor/outdoor (`setting`) and urban/rural (`environment`) classification heads on top of frozen CLIP image features. This is genuine training of the context classifier; it does not fine-tune CLIP itself or train the YOLO object detector and BLIP captioner. The app uses a trained head for any task that has a checkpoint and falls back to CLIP for other tasks.

### Train indoor/outdoor from a public dataset

The default data source is [IndoorOutdoorNet-20K](https://huggingface.co/datasets/prithivMLmods/IndoorOutdoorNet-20K), which publishes 19,998 indoor/outdoor labeled images under Apache-2.0. The dataset has a single split, so the training script makes a stratified 80/20 train/validation split. It contains no urban/rural labels; those predictions continue to use CLIP until you train the environment head on a separate dataset.

Install the optional training dependency and train from the project root:

```powershell
backend\.venv\Scripts\python.exe -m pip install -r backend\training-requirements.txt
backend\.venv\Scripts\python.exe backend\train_context.py --task setting --dataset-id prithivMLmods/IndoorOutdoorNet-20K --epochs 10
```

The first run downloads the dataset and pretrained CLIP weights. It caches CLIP features, reports validation accuracy, and writes the trained head to `backend/models/context_head.pt`. Restart the API after training to load it. The model will match its training data, so also evaluate it on new camera-view images before relying on it; a held-out split from the same source is not a guarantee of real-world accuracy.

### Train urban/rural with your own labeled images

Use photos that represent the same camera-view scenes you expect the app to analyze, split by source scene (not adjacent video frames):

```text
training-data/
  train/
    urban/
    rural/
  validation/
    urban/
    rural/
```

Then run:

```powershell
backend\.venv\Scripts\python.exe backend\train_context.py --task environment --data-dir training-data --epochs 10
```

The urban/rural task needs its own accurately labeled dataset; a rural/urban aerial-image dataset will not necessarily generalize to ground-level camera footage. Keep training data, downloaded datasets, and model weights out of Git unless their licenses and repository size make that appropriate. Training uses CUDA automatically when available; otherwise it runs on CPU.

## Deploy on Hugging Face Spaces

1. Create a new Space and select **Docker** as the SDK.
2. Upload or push this repository to the Space.
3. The Space builds the frontend and API into one container. It downloads the model weights when the container starts; allow the first build/start to finish before testing the camera.
4. Open the Space over HTTPS, grant camera permission, and select **Start camera**.

The demo uses public pretrained models and an unauthenticated inference endpoint. Anyone who can access a public Space can submit images for analysis, which consumes its compute. For a private or production service, add authentication and rate limiting and review the model licenses and hosting costs before sharing it broadly.

## Useful settings

The backend accepts `YOLO_MODEL`, `CLIP_MODEL`, and `BLIP_MODEL` environment variables to select the object, scene, and image-captioning models; `ALLOWED_ORIGINS` configures browser origins, and `PRELOAD_MODELS` warms the object and scene models at startup. For example, in PowerShell:

```powershell
$env:YOLO_MODEL = "yolo11n.pt"
$env:CLIP_MODEL = "openai/clip-vit-base-patch32"
$env:BLIP_MODEL = "Salesforce/blip-image-captioning-base"
```

To use a phone on the same network, serve the website over HTTPS for camera access. Do not expose the development server publicly.
