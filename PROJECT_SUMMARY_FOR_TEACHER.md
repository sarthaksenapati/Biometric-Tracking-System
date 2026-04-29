# Project Summary — Biometric Tracking System using AI & Computer Vision

**Students:** Prityanshu Yadav, Sarthak Senapati  
**Project Type:** B.Tech Final Year Project  
**Date:** March 2025

---

## 1. Intent & Objective

The project aims to build an **intelligent multi-modal biometric surveillance system** for campus-style environments. The core idea is to **identify and track individuals** using more than one biometric cue, so the system remains useful when a single modality fails (e.g., face occluded, far away, or poor lighting).

**Goals:**
- Identify people using **face**, **body (ReID)**, and **gait**
- Track people across video streams in real time
- Provide a foundation for future features: search by image, attribute-based search, multi-camera tracking, and a dashboard

---

## 2. Approaches Used

### 2.1 Multi-Modal Biometric Fusion

Instead of relying only on face recognition, the system combines three modalities:

| Modality | Purpose | When It Helps |
|----------|---------|---------------|
| **Face** | Primary identifier | Frontal/semi-profile views, good lighting |
| **Body (ReID)** | Secondary identifier | Face not visible, different angles |
| **Gait** | Tertiary identifier | Face occluded, walking from a distance |

### 2.2 Technical Approaches

1. **Embedding-based matching**
   - Each modality produces a feature vector (embedding).
   - Matching uses **cosine similarity** between live embeddings and stored embeddings.

2. **Weighted score fusion**
   - Weights are derived from **cross-similarity analysis** (how well each modality separates different people).
   - Face: ~0.95 (most discriminative)
   - Body: ~0.04
   - Gait: ~0.01
   - Weights are normalized so available modalities always sum to 1.

3. **Trust logic**
   - Face is required for a **trusted** identification.
   - Body + gait alone are not trusted, because cross-similarity between different people is high for these modalities.

4. **Temporal smoothing**
   - Gait uses a buffer of 10 frames for temporal averaging.
   - A prediction buffer of 7 frames reduces flicker in live identification.

### 2.3 Models & Frameworks

| Component | Technology |
|-----------|------------|
| Person detection | YOLOv8 (Ultralytics) + ByteTrack |
| Face recognition | InsightFace (buffalo_sc) or DeepFace (FaceNet) fallback |
| Body ReID | OSNet-x1.0 (MSMT17) or ResNet50 fallback |
| Gait | Custom silhouette-based temporal averaging (64×128) |
| Backend | FastAPI, Python 3.10+ |
| Deep learning | PyTorch, torchvision |
| Computer vision | OpenCV |

---

## 3. How People Are Differentiated

There is **no traditional user login or role-based access**. People are distinguished purely by **biometric identity**.

### 3.1 Registration

- Each person is registered with a **Person ID** (e.g., `prityanshu`).
- For each modality:
  - **Face:** 15 sample images
  - **Body:** 10 sample crops
  - **Gait:** 20–30 frames of walking
- Embeddings are stored as `{person_id}_{face|body|gait}.npy` in `embeddings_db/`.

### 3.2 Recognition

- Live video is processed to extract face, body, and gait embeddings.
- Each embedding is compared to all stored embeddings using cosine similarity.
- The **FusionEngine** combines modality scores with the weights above.
- The best match above a threshold is returned as the identified person; otherwise the result is `"Unknown"`.

### 3.3 Thresholds

- **With face:** threshold 0.45 (above typical face cross-similarity of ~0.15).
- **Without face:** threshold 0.99 (effectively disabled, so body/gait alone do not identify).

### 3.4 Summary of Differentiation

| Aspect | How It Works |
|--------|--------------|
| Identity | Person ID (name) assigned during registration |
| Storage | Per-person, per-modality embeddings in `.npy` files |
| Matching | Cosine similarity + weighted fusion |
| Output | Person ID or `"Unknown"` |
| Trust | Face must be available for trusted identification |

---

## 4. Achievements (Implemented So Far)

### Phase 1 — Person Detection
- YOLOv8-based real-time human detection
- ByteTrack for stable track IDs across frames
- Bounding box extraction for face, body, and gait crops

### Phase 2 — Face Recognition
- Face embeddings via InsightFace or DeepFace
- Registration (15 samples) and recognition pipelines
- Works for frontal and semi-profile views

### Phase 3 — Body Re-Identification
- OSNet-x1.0 (MSMT17) or ResNet50 fallback
- Registration (10 samples) and recognition
- Helps when face is not visible

### Phase 4 — Gait Recognition
- Silhouette-based gait feature extraction
- Temporal averaging of walking patterns
- Registration (20–30 frames) and recognition

### Phase 4+ — Multi-Modal Fusion & Live Tracking
- **FusionEngine** with data-driven weights
- **Matcher** for database lookup and fusion
- **LiveTracker** for real-time identification from webcam
- Trust logic: face required for trusted identification

### Supporting Tools
- `debug_scores.py`: Cross-similarity analysis and threshold suggestions
- `test_matcher.py`, `test_fusion.py`: Unit-style tests for matcher and fusion

---

## 5. Testing Methodology

The system is evaluated using:

- **Positive tests:** Registered person correctly identified
- **Negative tests:** Unregistered person correctly rejected
- **Robustness tests:** Different clothes, side view, low light

**Example results (from README):**

| Person | Score | Result |
|--------|-------|--------|
| Registered user | 0.85 – 0.95 | Correct |
| Other person 1 | 0.30 – 0.45 | Rejected |
| Other person 2 | 0.35 – 0.50 | Rejected |

---

## 6. Project Structure (High Level)

```
Final-Year-Project/
├── backend/          # Registration & recognition scripts (face, body, gait)
├── core/             # Tracker, Matcher, FusionEngine
├── models/           # Detector, FaceRecognizer, ReIDModel, GaitModel
├── utils/            # Config, embeddings, similarity
├── iot_stream/       # Camera reader & detection demo
├── run_tracker.py    # Entry point for live multi-modal tracking
├── debug_scores.py   # Cross-similarity analysis
└── embeddings_db/    # Stored embeddings (per person, per modality)
```

---

## 7. Future Work (Planned Phases)

| Phase | Feature | Status |
|-------|---------|--------|
| Phase 5 | Search by image | Planned |
| Phase 6 | Attribute-based search (clothing, height, etc.) | Planned |
| Phase 7 | Multi-camera tracking | Planned |
| Phase 8 | Dashboard & visualization | Planned |

---

## 8. Academic Relevance

The project applies concepts from:

- **Computer Vision** — Object detection, feature extraction, tracking
- **Machine Learning / Deep Learning** — Embeddings, similarity, fusion
- **Pattern Recognition** — Multi-modal biometrics, gait analysis
- **IoT Systems** — Camera streams, real-time processing
- **Surveillance Systems** — Identification, trust logic, thresholds

---

## 9. Summary for Quick Reference

| Topic | Summary |
|-------|---------|
| **Intent** | Multi-modal biometric identification and tracking for campus surveillance |
| **Approaches** | Face + body ReID + gait; embedding-based matching; weighted fusion; trust logic |
| **Differentiation** | Biometric identity via Person ID; no login/roles; embeddings stored per person per modality |
| **Achievements** | Phases 1–4 complete; fusion and live tracking implemented; debug and test tools |
| **Status** | Core system working; search, multi-camera, and dashboard planned |

---

*This document summarizes the current state of the project as of March 2025.*
