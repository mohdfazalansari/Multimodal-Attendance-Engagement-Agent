# Deploying Multimodal Attendance & Engagement Agent on Streamlit Community Cloud

This guide provides step-by-step instructions for deploying the **Multimodal Attendance & Engagement Agent** to **Streamlit Community Cloud** as a 100% free, public portfolio demonstration.

---

## 1. Deployment Architecture Overview

The production project employs an offline-first architecture with a FastAPI backend, Async SQLAlchemy, SQLite, React 19 frontend, and local GPU acceleration (NVIDIA CUDA). 

For **Streamlit Community Cloud**, an isolated serverless entry point (`streamlit_app.py`) is provided. This entry point interacts directly with existing backend service modules (`AttendanceService`, `SessionService`, `AnalyticsService`, `SuggestionService`, `StudentService`, and `MultimodalPipeline`) without requiring an independent FastAPI server process or paid cloud infrastructure.

```
                           +----------------------------------------+
                           |   Streamlit Community Cloud (Linux)    |
                           |       URL: <app-name>.streamlit.app    |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |            streamlit_app.py            |
                           |   (Cached ML models, ThreadPool Async) |
                           +-------------------+--------------------+
                                               |
                   +---------------------------+---------------------------+
                   |                           |                           |
                   v                           v                           v
         [ SQLite Local DB ]         [ Computer Vision ]         [ AI Suggestion Engine ]
       - Auto-seeded CS101 demo    - Frame-sampled CPU YOLO/HOG - Deterministic rule engine
       - In-memory / ephemeral     - solvePnP 3D Head Pose      - Observable evidence citations
       - Zero persistent uploads   - Aggregate RMS Audio VAD    - Urgency classification
```

---

## 2. Repository Deployment Files

The following deployment artifacts are configured in the repository root:

| File | Purpose |
|---|---|
| `streamlit_app.py` | Complete Streamlit entry point containing all 8 dashboard sections. |
| `requirements.txt` | Cloud-optimized dependencies targeting **Python 3.12** on Linux. Uses `opencv-python-headless` and pinned `numpy<2.0.0` for maximum compatibility. |
| `packages.txt` | Debian Linux packages (`ffmpeg`, `libgl1`, `libglib2.0-0`) enabling audio extraction and image decoding. |
| `.streamlit/config.toml` | UI theme configuration (modern dark blue/green palette) and file upload limit bounds (25 MB). |
| `.gitignore` | Enforces strict privacy by preventing commits of `.db` databases, `.npy` biometric vectors, `.mp4` uploads, or secrets. |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Push Changes to GitHub
Ensure your repository is pushed to your personal GitHub account:
```bash
git add .gitignore requirements.txt packages.txt .streamlit/config.toml streamlit_app.py STREAMLIT_DEPLOYMENT.md
git commit -m "feat: Add Streamlit Community Cloud deployment configuration and entry point"
git push origin main
```

> **Privacy Verification Check**: Confirm that `git status` shows zero private databases (`.db`), biometric templates (`.npy`), or `.env` files being tracked.

### Step 2: Open Streamlit Community Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your GitHub account.

### Step 3: Configure New Application
Click **"Create app"** (or **"New app"**) and enter:
- **Repository**: `your-username/Multimodal-Attendance-Engagement-Agent`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`
- **App URL**: Choose a custom subdomain (e.g. `multimodal-attendance-agent.streamlit.app`)

### Step 4: Specify Python Version (Critical)
1. Click **"Advanced settings..."** before deploying.
2. Under **Python version**, select **3.12**.
   > *Why Python 3.12?* Python 3.12 provides official pre-built Linux wheels for PyTorch, OpenCV, SciPy, and NumPy `< 2.0.0`. Python 3.13 is newer and may trigger compilation delays or wheel incompatibilities on Debian headless containers.
3. Keep Secrets empty (the app uses zero paid APIs and requires zero cloud keys).
4. Click **"Deploy!"**.

### Step 5: Verification & Cold-Boot
Streamlit will clone your repository, install Debian packages from `packages.txt`, install dependencies from `requirements.txt`, and boot the app.
- On first launch, `streamlit_app.py` automatically initializes SQLite and populates the rich Computer Science 101 demonstration cohorts, attendance logs, and observable engagement telemetry.

---

## 4. Local Execution & Testing

To test the Streamlit deployment locally before deploying to the cloud:

```bash
# 1. Run backend tests to verify core logic
cd backend
python -m pytest tests/ -v
cd ..

# 2. Verify frontend production build remains functional
cd frontend
npm run build
cd ..

# 3. Launch the Streamlit application locally
python -m streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 5. Feature Comparison: Local vs. Cloud Deployment

| Feature | Local Production Environment | Streamlit Community Cloud Demo |
|---|---|---|
| **Architecture** | FastAPI Backend + React 19 Frontend | Native Streamlit Unified Interface |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU (CUDA Enabled) | Shared Cloud CPU (Bounded Inference) |
| **Live Webcam Processing** | Real-time 30 FPS video streaming | Disabled (Cloud browsers lack direct hardware video access) |
| **Recorded Video Ingestion** | Full length 60-minute multi-camera lectures | Bounded Real Pipeline (clips ≤ 15MB) + Simulated Demo Mode |
| **Person Detection** | YOLOv8 nano via GPU / PyTorch | YOLOv8 nano / OpenCV HOG fallback via CPU |
| **Head Orientation** | `solvePnP` 3D perspective Euler angles | `solvePnP` 3D perspective Euler angles |
| **Voice Activity** | Aggregated RMS Energy & Zero-Crossing VAD | Aggregated RMS Energy & Zero-Crossing VAD via FFmpeg |
| **Biometric Matching** | FaceNet 512-dim cosine similarity | FaceNet 512-dim cosine similarity |
| **Pedagogical AI** | Local Ollama LLM (`llama3.2`) + Rule Engine | Deterministic Rule Engine + Observable Evidence Citations |
| **Privacy Model** | 100% Offline on localhost | Ephemeral Media (wiped immediately after analysis) |
| **Data Storage** | Persistent local SQLite & `.npy` files | Ephemeral SQLite with Reset/Re-seed capability |

---

## 6. Resource & Safety Guardrails

Streamlit Community Cloud provides 1 vCPU and ~1 GB RAM in its free tier. The following guardrails are enforced:
1. **Model Weight Caching (`@st.cache_resource`)**: YOLOv8 weights and the `MultimodalPipeline` are initialized only once per server process.
2. **Video Upload Limits**: Uploads are capped at 15 MB / 30 seconds to prevent server out-of-memory (OOM) termination.
3. **Frame Sampling**: When analyzing video, frames are sampled at ~1-2 FPS with an upper bound of 60 processed frames.
4. **Simulated High-Volume Mode**: If users wish to observe full 50-minute multi-window classroom analytics without consuming CPU time, the **Simulated Classroom Demo** provides an instant, transparently labeled walkthrough.
5. **Ephemeral Media Cleanup**: Any temporary video file written to `/tmp` is deleted immediately in a `finally:` block upon analysis completion, cancellation, or error.
