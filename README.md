# EyesSpies

A browser-first computer-vision prototype. The React app reads a camera frame about once per second and sends it to a local Python API. The API returns object detections from YOLO and indoor/outdoor plus urban/rural estimates from CLIP.

## What you need

- Node.js 20 or newer
- Python 3.10 or newer
- A webcam (or phone camera if you open the site over HTTPS)

## Start the Python API

Open PowerShell in the project folder:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

The API listens at `http://localhost:8000`. On the first analysis request, it downloads the YOLO and CLIP model weights; this can take a while and requires an internet connection. It runs on CPU by default.

## Start the website

Open a second PowerShell window in the project folder:

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`) and allow camera access. Keep the API terminal running. Vite forwards `/api` requests to the API.

## How it works

- `src/` contains the browser UI and camera capture code.
- `backend/app.py` serves `/api/health` and `/api/analyze`.
- The analyzer returns bounding boxes and confidence scores, plus one label for each of two separate context questions.
- Frames are processed in memory and are not written to disk.

The context scores are relative scores between the two labels in each category, not calibrated probabilities. This is a learning prototype: evaluate it on your own scenes and data before relying on its predictions.

## Useful settings

The backend accepts `YOLO_MODEL` and `CLIP_MODEL` environment variables to select model IDs, and `ALLOWED_ORIGINS` to configure browser origins. For example, in PowerShell:

```powershell
$env:YOLO_MODEL = "yolo11n.pt"
$env:CLIP_MODEL = "openai/clip-vit-base-patch32"
```

To use a phone on the same network, the website must be served over HTTPS for camera access, and the API URL/CORS settings must be configured for that deployment. Do not expose the development server or unauthenticated API publicly.
