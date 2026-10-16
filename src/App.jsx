import { useCallback, useEffect, useRef, useState } from "react";

const ANALYSIS_INTERVAL_MS = 900;

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imageInputRef = useRef(null);
  const streamRef = useRef(null);
  const loopRef = useRef(false);
  const confidenceRef = useRef(0.35);
  const [cameraOn, setCameraOn] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.35);
  const [imagePreview, setImagePreview] = useState("");
  const [cameraError, setCameraError] = useState("");
  const [apiError, setApiError] = useState("");
  const [status, setStatus] = useState("Ready");
  const [result, setResult] = useState(null);
  const [imageCaption, setImageCaption] = useState("");
  const [captionLoading, setCaptionLoading] = useState(false);

  const analyzeImage = useCallback(async (blob, includeCaption = false) => {
    try {
      setStatus("Analyzing image");
      const params = new URLSearchParams({
        confidence_threshold: confidenceRef.current.toString(),
      });
      if (includeCaption) params.set("include_caption", "true");
      const response = await fetch(`/api/analyze?${params}`, {
        method: "POST",
        headers: { "Content-Type": blob.type },
        body: blob,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Analysis failed (${response.status}).`);
      }

      const analysis = await response.json();
      setResult(analysis);
      if (includeCaption) setImageCaption(analysis.caption || "");
      setApiError("");
      setStatus("Analysis complete");
    } catch (error) {
      setApiError(error.message || "Could not reach the analysis API.");
      setStatus("Analysis unavailable");
    }
  }, []);

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setApiError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setApiError("Choose an image smaller than 5 MB.");
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    setImagePreview((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      return previewUrl;
    });
    setResult(null);
    setImageCaption("");
    setApiError("");
    setCaptionLoading(true);
    await analyzeImage(file, true);
    setCaptionLoading(false);
  };

  const stopCamera = useCallback(() => {
    loopRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setStatus("Camera stopped");
  }, []);

  const analyzeFrames = useCallback(async () => {
    while (loopRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise((resolve) => window.setTimeout(resolve, 150));
        continue;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.8),
      );
      if (!blob) {
        setApiError("Could not capture a camera frame.");
        setStatus("Analysis unavailable");
      } else {
        await analyzeImage(blob);
        if (loopRef.current) setStatus("Live");
      }

      await new Promise((resolve) => window.setTimeout(resolve, ANALYSIS_INTERVAL_MS));
    }
  }, [analyzeImage]);

  const startCamera = async () => {
    setCameraError("");
    setApiError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("Camera access needs a secure browser context (localhost or HTTPS).");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      loopRef.current = true;
      setCameraOn(true);
      setStatus("Starting");
      analyzeFrames();
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraError(
        error.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access in your browser settings."
          : error.message || "Could not start the camera.",
      );
      setStatus("Camera unavailable");
    }
  };

  useEffect(() => () => {
    loopRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  useEffect(() => () => {
    if (imagePreview) URL.revokeObjectURL(imagePreview);
  }, [imagePreview]);

  const detections = result?.detections ?? [];

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="EyesSpies home">
          <span className="brand-mark" aria-hidden="true">E</span>
          <span>eyes<span className="brand-light">spies</span></span>
        </a>
        <span className="topbar-note"><span className="live-dot" /> COMPUTER VISION LAB</span>
      </header>

      <section className="intro">
        <p className="eyebrow">REAL-TIME SCENE INTELLIGENCE</p>
        <h1>See what’s <span>around you.</span></h1>
        <p className="intro-copy">
          Live object detection with a little more context. Point your camera at a scene to get started.
        </p>
      </section>

      <section className="workspace" aria-label="Live camera analysis">
        <div className="camera-card">
          <div className="camera-heading">
            <div>
              <p className="card-kicker">CAMERA FEED</p>
              <h2>Live view</h2>
            </div>
            <span className={`status-pill ${cameraOn ? "is-live" : ""}`}>
              <span className="status-dot" /> {status}
            </span>
          </div>

          <div className="video-stage">
            <video ref={videoRef} className={cameraOn ? "visible" : ""} playsInline muted />
            {cameraOn && result && videoRef.current && (
              <div className="detection-layer" aria-hidden="true">
                {detections.map((detection, index) => {
                  const [x1, y1, x2, y2] = detection.box;
                  return (
                    <div
                      className="detection-box"
                      key={`${detection.label}-${index}`}
                      style={{
                        left: `${(x1 / videoRef.current.videoWidth) * 100}%`,
                        top: `${(y1 / videoRef.current.videoHeight) * 100}%`,
                        width: `${((x2 - x1) / videoRef.current.videoWidth) * 100}%`,
                        height: `${((y2 - y1) / videoRef.current.videoHeight) * 100}%`,
                      }}
                    >
                      <span>{detection.label} · {formatPercent(detection.confidence)}</span>
                    </div>
                  );
                })}
              </div>
            )}
            {!cameraOn && imagePreview && (
              <img className="uploaded-image" src={imagePreview} alt="Image selected for analysis" />
            )}
            {!cameraOn && !imagePreview && (
              <div className="camera-placeholder">
                <div className="camera-icon" aria-hidden="true">◎</div>
                <strong>Your camera or selected image will appear here</strong>
                <span>Choose a sample photo to test the vision models.</span>
              </div>
            )}
          </div>
          <canvas ref={canvasRef} className="hidden-canvas" />
          <input
            ref={imageInputRef}
            className="hidden-canvas"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleImageUpload}
          />
          {cameraError && <p className="error-message">{cameraError}</p>}
          {apiError && <p className="error-message">{apiError}</p>}
          <div className="camera-footer">
            <span>{cameraOn ? "Frames analyzed about once per second" : "Try a sample image or live camera"}</span>
            {cameraOn ? (
              <button className="button button-stop" onClick={stopCamera}>Stop camera</button>
            ) : (
              <div className="camera-actions">
                <button
                  className="button button-upload"
                  onClick={() => imageInputRef.current?.click()}
                >
                  Choose image
                </button>
                <button className="button button-start" onClick={startCamera}>Start camera <span aria-hidden="true">↗</span></button>
              </div>
            )}
          </div>
          <div className="confidence-control">
            <label htmlFor="confidence-threshold">
              Detection confidence
              <strong>{formatPercent(confidenceThreshold)}</strong>
            </label>
            <input
              id="confidence-threshold"
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={confidenceThreshold}
              onChange={(event) => {
                const value = Number(event.target.value);
                confidenceRef.current = value;
                setConfidenceThreshold(value);
              }}
              aria-describedby="confidence-help"
            />
            <span id="confidence-help" className="confidence-help">
              Higher values hide uncertain detections; changes apply to the next frame.
            </span>
          </div>
        </div>

        <aside className="results-column">
          {!cameraOn && (imagePreview || captionLoading) && (
            <section className="result-card caption-card" aria-live="polite">
              <div className="card-kicker">IMAGE ANALYSIS</div>
              <h2>What’s happening?</h2>
              <p className="caption-text">
                {captionLoading
                  ? "Generating a description of your image…"
                  : imageCaption || "The image description is not available."}
              </p>
              <p className="muted-note">
                Generated by an image-captioning model; verify important details yourself.
              </p>
            </section>
          )}
          <section className="result-card">
            <div className="card-kicker">SCENE CONTEXT</div>
            <h2>Where are we?</h2>
            <div className="context-list">
              <ContextScore
                title="Setting"
                value={result?.context?.setting}
                confidence={result?.context?.setting_confidence}
                options={["Indoor", "Outdoor"]}
              />
              <ContextScore
                title="Environment"
                value={result?.context?.environment}
                confidence={result?.context?.environment_confidence}
                options={["Urban", "Rural"]}
              />
            </div>
            <p className="muted-note">Scene labels are model estimates, not ground truth.</p>
          </section>

          <section className="result-card objects-card">
            <div className="objects-heading">
              <div>
                <div className="card-kicker">OBJECT DETECTION</div>
                <h2>In view</h2>
              </div>
              <span className="count-badge">{detections.length}</span>
            </div>
            {detections.length ? (
              <ul className="object-list">
                {detections.slice(0, 8).map((detection, index) => (
                  <li key={`${detection.label}-${index}`}>
                    <span className="object-indicator" />
                    <span>{detection.label}</span>
                    <span className="object-confidence">{formatPercent(detection.confidence)}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty-state">
                <span aria-hidden="true">⌁</span>
                <p>{cameraOn ? "No objects detected yet." : "Start your camera to detect objects."}</p>
              </div>
            )}
          </section>

          <p className="privacy-note"><span aria-hidden="true">◈</span> Camera frames are sent to the server for analysis and aren’t saved by this app.</p>
        </aside>
      </section>

      <footer className="page-footer">
        <span>BUILT FOR EXPLORATION</span>
        <span>EyesSpies · Vision prototype</span>
      </footer>
    </main>
  );
}

function ContextScore({ title, value, confidence, options }) {
  return (
    <div className="context-item">
      <div className="context-title">
        <span>{title}</span>
        <strong>{value || "Waiting for camera"}</strong>
      </div>
      <div className="score-track" aria-label={value ? `${value}: ${formatPercent(confidence)}` : "No prediction yet"}>
        <span
          className={`score-fill ${value ? "" : "score-empty"}`}
          style={{ width: value ? `${Math.max(confidence * 100, 4)}%` : "0%" }}
        />
      </div>
      <div className="score-labels">
        <span>{options[0]}</span><span>{options[1]}</span>
      </div>
    </div>
  );
}
