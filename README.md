# 🧠 Multi-Modal Biometric Tracking System

An AI-powered biometric surveillance system that identifies and tracks individuals across campus environments using **face recognition**, **body re-identification**, and **gait analysis**. Built for real-time multi-camera tracking with a web-based monitoring dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)

---

## 🎯 Project Overview

This system addresses the limitations of single-modal biometric systems by combining **three complementary modalities** for robust person identification:

- **Face Recognition** — Primary identifier using facial features
- **Body Re-Identification (ReID)** — Secondary identifier using body structure and clothing
- **Gait Recognition** — Tertiary identifier using walking patterns

---

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Input  │    │  Person Detection│    │ Feature Extract │
│   (IoT Streams) │───▶│    (YOLOv8)     │───▶│   (Multi-Modal) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Feature Fusion │    │  Identity Match  │    │  Multi-Camera   │
│   & Matching    │◀───│   (Database)    │───▶│    Tracking     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │  Dashboard &    │
                                               │   API Output    │
                                               └─────────────────┘
```

### Key Components

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Detection** | YOLOv8 + ByteTrack | Real-time person detection & stable tracking |
| **Face Model** | InsightFace / DeepFace | 512D face embeddings |
| **Body ReID** | Custom OSNet-x1.0 | 2048D body embeddings |
| **Gait Model** | Silhouette-based | 64×128 temporal features |
| **Fusion Engine** | Weighted scoring | Face: 95%, Body: 4%, Gait: 1% |
| **Database** | PostgreSQL 16+ | Embeddings, events, detections |
| **Cache** | Redis 7+ | Fast embedding lookups |
| **Backend** | FastAPI + Uvicorn | REST API |
| **Dashboard** | Streamlit | Web monitoring interface |

---

## 🚀 Features

### Detection & Tracking
- Real-time human detection using YOLOv8
- Stable tracking with ByteTrack algorithm
- Multi-camera support with identity handoff
- Temporal smoothing to reduce identification flicker

### Multi-Modal Biometric Recognition
- Face, body, and gait recognition with fusion-based matching
- Trust system: face required for confident identification
- Unknown person detection and storage
- Cosine similarity matching with configurable thresholds

### Database & Storage
- **PostgreSQL** as primary storage (replaces `.npy` files)
- **Redis** caching layer for fast embedding lookups
- **Automatic fallback** to file-based storage when DB is unavailable
- Support for multiple exemplars per person

### Monitoring Dashboard
- Real-time active person display
- Event logging (sightings, handoffs)
- Person search across current session and history
- Camera status monitoring with 7-day history retention

### Administration
- Per-modality registration scripts
- Embedding management and renaming utilities
- Cross-similarity analysis for threshold tuning
- Debug tools for performance evaluation

---

## 🧪 Testing & Performance

### Example Results

| Test Case | Modality | Score | Result |
|-----------|----------|-------|--------|
| Registered User (Face) | Face | 0.92 | ✅ Correct |
| Registered User (Body) | Body | 0.78 | ✅ Correct |
| Unregistered Person | Face | 0.35 | ❌ Rejected |
| Different Clothing | Body | 0.65 | ✅ Robust |
| Occluded Face | Gait | 0.71 | ✅ Fallback |

### Performance Metrics
- **Accuracy**: >90% for registered persons with face available
- **False Positive Rate**: <5% with proper thresholds
- **Real-time Processing**: 15–20 FPS on GPU, 5–8 FPS on CPU
- **Memory Usage**: ~2GB for models + embeddings

---

## 🧰 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+, TorchVision |
| **Computer Vision** | OpenCV 4.8+, Ultralytics YOLOv8 |
| **Face Recognition** | InsightFace, DeepFace (FaceNet) |
| **Body ReID** | Custom OSNet-x1.0 (MSMT17 pretrained) |
| **Gait Analysis** | Custom silhouette extraction |
| **Database** | PostgreSQL 16+ (primary), NumPy files (fallback) |
| **Caching** | Redis 7+ |
| **Message Queue** | Redis Streams / RabbitMQ (optional) |
| **Backend** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Deployment** | Docker, Docker Compose, Render.com |
| **CI/CD** | GitHub Actions |

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended) or Intel Iris Xe
- Webcam or IP cameras

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/sarthaksenapati/Biometric-Tracking-System.git
cd Biometric-Tracking-System

# 2. Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python test.py
```

> **Models**: YOLOv8 and face models are downloaded automatically. OSNet weights (`osnet_x1_0_msmt17.pth`) are included in the repository.

---

## ▶️ Usage

### 1. Register a Person

```bash
python -m backend.register         # Face (captures 15 images)
python -m backend.register_body    # Body (captures 10 crops)
python -m backend.register_gait    # Gait (captures 20–30 frames walking)
```

### 2. Run Live Tracking

```bash
python run_tracker.py              # Single camera
python run_tracker_multi.py        # Multi-camera
```

### 3. Launch Dashboard

```bash
python -m streamlit run dashboard.py
```

### 4. Start Backend API

```bash
uvicorn backend.app:app --reload
```

### Camera Configuration

Edit `run_tracker_multi.py` to set your camera sources:

```python
sources = {
    0: 0,             # Webcam
    1: "rtsp://...",  # IP Camera
}
cam_locations = {
    0: "Entrance",
    1: "Parking Lot",
}
```

### Person Management

```bash
python -m utils.admin_controls rename OldName NewName  # Rename a person
python debug_scores.py                                  # Analyze cross-similarities
python GPU.py                                           # Check GPU availability
```

---

## 📁 Project Structure

```
biometric-tracking-system/
│
├── backend/                  # Registration & recognition scripts
│   ├── app.py                # FastAPI backend
│   ├── register.py           # Face registration
│   ├── recognize.py          # Face recognition
│   ├── register_body.py      # Body registration
│   ├── recognize_body.py     # Body recognition
│   ├── register_gait.py      # Gait registration
│   └── recognize_gait.py     # Gait recognition
│
├── core/                     # Core tracking logic
│   ├── tracker.py            # Single-camera tracker
│   ├── multi_tracker.py      # Multi-camera tracker
│   ├── matcher.py            # Database matching
│   └── fusion_engine.py      # Modality fusion
│
├── models/                   # AI model implementations
│   ├── detector.py           # YOLOv8 person detection
│   ├── face_model.py         # Face recognition
│   ├── reid_model.py         # Body ReID (OSNet)
│   └── gait_model.py         # Gait recognition
│
├── db/                       # PostgreSQL connection & models
├── cache/                    # Redis caching layer
├── task_queue/               # Message queue abstraction
├── utils/                    # Utilities & helpers
├── iot_stream/               # Camera interface
├── embeddings_db/            # Stored person embeddings (.npy)
├── deploy/                   # Cloud/edge deployment configs
│
├── run_tracker.py            # Single-camera entry point
├── run_tracker_multi.py      # Multi-camera entry point
├── dashboard.py              # Streamlit dashboard
├── migrate_to_db.py          # Migrate .npy files to PostgreSQL
├── debug_scores.py           # Cross-similarity analysis
├── GPU.py                    # GPU availability check
├── requirements.txt
├── runtime.txt               # Python version pinning (3.11.0)
├── docker-compose.yml
├── Dockerfile.tracker
├── Dockerfile.backend
├── Dockerfile.dashboard
└── README.md
```

---

## 🔧 Configuration

### `utils/config.py`

```python
# Detection
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5

# Fusion weights (data-driven)
FACE_WEIGHT = 0.95
BODY_WEIGHT = 0.04
GAIT_WEIGHT = 0.01

# Matching thresholds
FACE_THRESHOLD = 0.45
BODY_THRESHOLD = 0.99  # Effectively disabled without face
```

---

## 🐳 Docker Setup

```bash
cp .env.example .env       # Set up environment variables
docker-compose build        # Build all images
docker-compose up -d        # Start all services
docker-compose logs -f      # View logs
docker-compose down         # Stop all services
```

### Services

| Service | Description | Port |
|---------|-------------|------|
| **tracker** | Multi-camera tracker | 8502 |
| **backend** | FastAPI backend API | 8000 |
| **dashboard** | Streamlit dashboard | 8501 |
| **redis** | Redis cache & messaging | 6379 |

**Platform notes:**
- **Linux**: Use `docker-compose.linux.yml` override for camera access
- **Windows**: Use `docker-compose.windows.yml` or run tracker locally
- **Mac**: Run tracker locally, containerize backend + dashboard

---

## 🚀 Deployment

### Hybrid Architecture (Recommended)

The tracker requires local camera access, while the backend and dashboard run in the cloud:

```
LOCAL MACHINE
└── Tracker (run_tracker_multi.py)
    └── Camera access, local AI models
         │ HTTP/REST
         ▼
RENDER.COM
├── Backend API (FastAPI) — Port 8000
├── Dashboard (Streamlit)  — Port 8501
└── Redis Cache
```

### Deploy to Render.com

1. Push to GitHub (triggers CI/CD automatically)
2. Go to [Render.com](https://render.com) → **New +** → **Blueprint** → connect your repo
3. Add GitHub secrets: `RENDER_DEPLOY_HOOK_BACKEND`, `RENDER_DEPLOY_HOOK_DASHBOARD`
4. Update `config.py` with your Render URLs and run tracker locally:

```python
BACKEND_URL = "https://biometric-backend.onrender.com"
USE_REMOTE_BACKEND = True
```

```bash
python run_tracker_multi.py
```

### Database Migration

```bash
export USE_DATABASE=true
python migrate_to_db.py   # Migrate .npy files to PostgreSQL
```

---

## 🐛 Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA not available | Install CUDA toolkit + compatible PyTorch. Check with `python GPU.py` |
| Low detection accuracy | Ensure good lighting, adjust `CONF_THRESHOLD` in `config.py` |
| Dashboard not loading | Check `tracker_state.json` exists; start tracker first |
| Import errors | Reinstall: `pip install -r requirements.txt`, verify Python 3.10+ |

---

## 🔮 Planned Features

| Phase | Feature |
|-------|---------|
| Phase 5 | Image-based person search |
| Phase 6 | Attribute-based filtering (clothing color, height, etc.) |
| Phase 7 | Full campus map visualization |
| Phase 8 | Advanced analytics and reporting |

---

## 📚 Academic Context

This project demonstrates concepts from Computer Vision, Machine Learning, Deep Learning, Pattern Recognition, IoT Systems, and Multi-modal Biometric Authentication.

**Key algorithms:** YOLOv8, ByteTrack, OSNet, Cosine Similarity, Weighted Fusion

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

Follow PEP 8, add docstrings to new functions, and test with `python test.py`.

---

## 👨‍💻 Authors

**Prityanshu Yadav** — Project Lead, Core Development  
B.Tech Final Year, Department of Computer Science & Engineering  
📧 prityanshu.yadav@email.com · [GitHub](https://github.com/prityanshu-yadav) · [LinkedIn](https://linkedin.com/in/prityanshu-yadav)

**Sarthak Senapati** — Co-Developer, Development & Testing  
Department of Computer Science & Engineering  
📧 sarthaksenapati566@gmail.com · [GitHub](https://github.com/sarthaksenapati) · [LinkedIn](https://linkedin.com/in/sarthaksenapati)

---

## ⭐ Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [InsightFace](https://github.com/deepinsight/insightface) for face recognition
- [DeepFace](https://github.com/serengil/deepface) for fallback face model
- [PyTorch](https://pytorch.org/) community
- [OpenCV](https://opencv.org/) team

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

*Developed for academic and research purposes. Ensure compliance with privacy laws and ethical guidelines when deploying in real environments.*

---

*⭐ Star this repository if you find it useful!*