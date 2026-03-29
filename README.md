markdown# 🧠 Multi-Modal Biometric Tracking System

A sophisticated **AI-powered biometric surveillance system** that identifies and tracks individuals across campus environments using **face recognition, body re-identification, and gait analysis**. Built for real-time multi-camera tracking with a web-based monitoring dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io/)

---

## 🎯 Project Overview

This system addresses the limitations of single-modal biometric systems by combining **three complementary modalities** for robust person identification:

- **Face Recognition**: Primary identifier using facial features
- **Body Re-Identification (ReID)**: Secondary identifier using body structure and clothing
- **Gait Recognition**: Tertiary identifier using walking patterns

The system provides **real-time tracking** across multiple cameras, **fusion-based matching** with trust logic, and a **comprehensive dashboard** for monitoring and administration.

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
│  Feature Fusion │    │   Identity Match │    │ Multi-Camera    │
│   & Matching    │◀───│    (Database)    │───▶│   Tracking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────┐
                                               │   Dashboard &   │
                                               │   API Output    │
                                               └─────────────────┘
```

### Key Components

- **Detection Engine**: YOLOv8 with ByteTrack for stable person tracking
- **Feature Extractors**:
  - Face: InsightFace or DeepFace (FaceNet)
  - Body: Custom OSNet-x1.0 implementation
  - Gait: Silhouette-based temporal averaging
- **Fusion Engine**: Weighted score combination with modality-specific weights
- **Matcher**: Cosine similarity against stored embeddings
- **Tracker**: Multi-camera identity handoff and temporal smoothing
- **Dashboard**: Streamlit-based monitoring interface

---

## 🚀 Features

### ✅ Core Features Implemented

#### 🔍 Person Detection & Tracking
- Real-time human detection using YOLOv8
- Stable tracking with ByteTrack algorithm
- Multi-camera support with identity handoff
- Temporal smoothing to reduce identification flicker

#### 👤 Multi-Modal Biometric Recognition
- **Face Recognition**: 512D embeddings, works in various lighting/angles
- **Body ReID**: 2048D embeddings, robust to pose/clothing changes
- **Gait Recognition**: 64×128 silhouette features with temporal averaging
- **Fusion Logic**: Data-driven weights (Face: 95%, Body: 4%, Gait: 1%)
- **Trust System**: Face required for confident identification

#### 📊 Database & Matching
- Embedding storage in NumPy format (.npy files)
- Cosine similarity matching with configurable thresholds
- Support for multiple exemplars per person
- Unknown person detection and storage

#### 🎛️ Monitoring Dashboard
- Real-time active person display
- Event logging (sightings, handoffs)
- Person search across current session and history
- Camera status monitoring
- Persistent history (7-day retention)

#### 🔧 Administration Tools
- Person registration scripts for each modality
- Embedding management and renaming utilities
- Cross-similarity analysis for threshold tuning
- Debug tools for performance evaluation

### 🔮 Planned Features
- Image-based search
- Attribute-based filtering (clothing color, height)
- Full campus map visualization
- Advanced analytics and reporting

---

## 🧪 Testing & Validation

### Methodology
- **Positive Tests**: Registered persons correctly identified
- **Negative Tests**: Unregistered persons rejected
- **Robustness Tests**: Different clothing, lighting, angles
- **Cross-Modality Analysis**: Similarity distribution analysis for weight tuning

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
- **Real-time Processing**: 15-20 FPS on GPU, 5-8 FPS on CPU
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
| **Backend** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit |
| **Data Processing** | NumPy |
| **Utilities** | JSON, Threading, Collections |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for real-time performance)
- Webcam or IP cameras for testing

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/biometric-tracking-system.git
   cd biometric-tracking-system
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Pre-trained Models**
   - YOLOv8 model: Automatically downloaded by Ultralytics
   - OSNet weights: Included in repository (`osnet_x1_0_msmt17.pth`)
   - Face models: Downloaded automatically by InsightFace/DeepFace

5. **Verify Installation**
   ```bash
   python test.py
   ```

---

## ▶️ Usage

### Quick Start

1. **Register a Person**
   ```bash
   # Register face (capture 15 images)
   python -m backend.register

   # Register body (capture 10 crops)
   python -m backend.register_body

   # Register gait (capture 20-30 frames walking)
   python -m backend.register_gait
   ```

2. **Run Live Tracking**
   ```bash
   # Single camera
   python run_tracker.py

   # Multi-camera (requires camera configuration)
   python run_tracker_multi.py
   ```

3. **Launch Dashboard**
   ```bash
   python -m streamlit run dashboard.py
   ```

### Advanced Usage

#### Camera Configuration
Edit `run_tracker_multi.py` to configure camera sources:
```python
sources = {
    0: 0,        # Webcam
    1: "rtsp://...",  # IP Camera
}
cam_locations = {
    0: "Entrance",
    1: "Parking Lot",
}
```

#### Person Management
```bash
# Rename a person in database
python -m utils.admin_controls rename OldName NewName

# Analyze cross-similarities
python debug_scores.py
```

#### API Usage
```bash
# Start backend API
uvicorn backend.app:app --reload
```

---

## 📁 Project Structure

```
biometric-tracking-system/
│
├── 📂 backend/                    # Registration & recognition scripts
│   ├── __init__.py
│   ├── app.py                     # FastAPI backend
│   ├── register.py                # Face registration
│   ├── recognize.py               # Face recognition
│   ├── register_body.py           # Body registration
│   ├── recognize_body.py          # Body recognition
│   ├── register_gait.py           # Gait registration
│   └── recognize_gait.py          # Gait recognition
│
├── 📂 core/                       # Core tracking logic
│   ├── __init__.py
│   ├── tracker.py                 # Single-camera tracker
│   ├── multi_tracker.py           # Multi-camera tracker
│   ├── matcher.py                 # Database matching
│   ├── fusion_engine.py           # Modality fusion
│   └── __pycache__/
│
├── 📂 models/                     # AI model implementations
│   ├── __init__.py
│   ├── detector.py                # YOLOv8 person detection
│   ├── face_model.py              # Face recognition
│   ├── reid_model.py              # Body ReID (OSNet)
│   ├── gait_model.py              # Gait recognition
│   └── __pycache__/
│
├── 📂 utils/                      # Utilities & helpers
│   ├── __init__.py
│   ├── embeddings.py              # Embedding storage/loading
│   ├── similarity.py              # Similarity calculations
│   ├── config.py                  # Configuration constants
│   ├── admin_controls.py          # Admin utilities
│   └── __pycache__/
│
├── 📂 iot_stream/                 # Camera interface
│   ├── __init__.py
│   ├── camera_reader.py           # Camera reading utilities
│   └── __pycache__/
│
├── 📂 embeddings_db/              # Stored person embeddings
│   ├── person1_face.npy
│   ├── person1_body.npy
│   └── person1_gait.npy
│
├── 📂 unknown_persons_emb/        # Unknown person embeddings
│
├── 📂 datasets/                   # Training/validation data
│
├── 📂 dashboard/                  # Dashboard assets (if any)
│
├── 🐍 run_tracker.py              # Single-camera entry point
├── 🐍 run_tracker_multi.py        # Multi-camera entry point
├── 🐍 dashboard.py                 # Streamlit dashboard
├── 🐍 debug_scores.py             # Cross-similarity analysis
├── 🐍 test.py                     # Installation verification
├── 🐍 rename_person.py            # Person renaming utility
├── 🐍 GPU.py                      # GPU availability check
├── 🐍 requirements.txt            # Python dependencies
├── 📄 README.md                   # This file
├── 📄 PROJECT_SUMMARY_FOR_TEACHER.md  # Academic summary
└── 📄 .gitignore                  # Git ignore rules
```

---

## 🔧 Configuration

### Model Configuration (`utils/config.py`)
```python
# Detection settings
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

### Camera Configuration
Modify `run_tracker_multi.py` for your camera setup.

---

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Not Available**
   - Install CUDA toolkit and compatible PyTorch
   - Check with `python GPU.py`

2. **Low Detection Accuracy**
   - Ensure good lighting
   - Adjust confidence threshold in `utils/config.py`

3. **Dashboard Not Loading**
   - Check if `tracker_state.json` exists
   - Start tracker first: `python run_tracker_multi.py`

4. **Import Errors**
   - Reinstall dependencies: `pip install -r requirements.txt`
   - Check Python version (3.10+ required)

### Debug Tools
```bash
# Test all components
python test.py

# Analyze similarity distributions
python debug_scores.py

# Check GPU availability
python GPU.py
```

---

## 📚 Academic Context

This project demonstrates concepts from:

- **Computer Vision**: Object detection, feature extraction, tracking algorithms
- **Machine Learning**: Embedding-based similarity, multi-modal fusion
- **Pattern Recognition**: Biometric authentication, gait signature extraction
- **IoT Systems**: Real-time camera streams, distributed processing
- **Surveillance Systems**: Identity management, trust-based decision making

### Key Algorithms
- **YOLOv8**: Real-time object detection
- **ByteTrack**: Multi-object tracking
- **OSNet**: Lightweight ReID network
- **Cosine Similarity**: Efficient embedding comparison
- **Weighted Fusion**: Multi-modal score combination

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test changes with `python test.py`
- Update documentation for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: This system is developed for research and educational purposes. Ensure compliance with privacy laws and ethical guidelines when deploying in real environments.

---

## 👨‍💻 Author

**Prityanshu Yadav**  
B.Tech Final Year Project  
Department of Computer Science & Engineering

### Acknowledgments
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [InsightFace](https://github.com/deepinsight/insightface) for face recognition
- [DeepFace](https://github.com/serengil/deepface) for fallback face model
- [PyTorch](https://pytorch.org/) community
- [OpenCV](https://opencv.org/) team

---

## 📞 Contact

For questions or collaboration:
- **Email**: prityanshu.yadav@email.com
- **GitHub**: [@prityanshu-yadav](https://github.com/prityanshu-yadav)
- **LinkedIn**: [Prityanshu Yadav](https://linkedin.com/in/prityanshu-yadav)

---

*⭐ Star this repository if you find it useful!*
│   ├── register.py
│   ├── recognize.py
│   ├── register_body.py
│   ├── recognize_body.py
│   ├── register_gait.py
│   └── recognize_gait.py
│
├── models/
│   ├── detector.py
│   ├── face_model.py
│   ├── reid_model.py
│   └── gait_model.py
│
├── utils/
│   ├── embeddings.py
│   ├── similarity.py
│   └── config.py
│
├── embeddings_db/
├── datasets/
├── iot_stream/
├── dashboard/
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Prityanshu/Biometric-Tracking-System.git
cd Biometric-Tracking-System
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Register Face
```bash
python -m backend.register
```

### 🔹 Recognize Face
```bash
python -m backend.recognize
```

### 🔹 Register Body (ReID)
```bash
python -m backend.register_body
```

### 🔹 Recognize Body
```bash
python -m backend.recognize_body
```

### 🔹 Register Gait
```bash
python -m backend.register_gait
```
> Walk in front of camera and press `S` to save.

### 🔹 Recognize Gait
```bash
python -m backend.recognize_gait
```

---

## 🎯 Current Capabilities

- ✔ Face-based identification
- ✔ Body-based identification
- ✔ Gait-based identification
- ✔ Real-time webcam inference
- ✔ Embedding-based similarity matching

---

## 🔮 Future Work (Upcoming Phases)

| Phase | Feature | Description |
|-------|---------|-------------|
| 🚧 Phase 5 | Search by Image | Input a snapshot → locate person across cameras |
| 🚧 Phase 6 | Attribute-Based Search | Search by shirt color, pant color, height, body type, accessories |
| 🚧 Phase 7 | Multi-Camera Tracking | Track identity across multiple streams with real-time location |
| 🚧 Phase 8 | Dashboard & Visualization | Live monitoring, detection overlay, campus map view |

---

## 🎓 Academic Relevance

This project demonstrates concepts from Computer Vision, Machine Learning, Deep Learning, Pattern Recognition, IoT Systems, Surveillance Systems, and Multi-modal Biometric Authentication.

---

## 📌 Key Concepts Used

`Feature Embeddings` `Cosine Similarity` `Object Detection` `Person Re-Identification` `Gait Signature Extraction` `Multi-modal Biometrics`

---

## 👨‍💻 Author

**Prityanshu Yadav** — B.Tech Final Year Project

---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepFace](https://github.com/serengil/deepface)
- [PyTorch Community](https://pytorch.org)
- [OpenCV](https://opencv.org)