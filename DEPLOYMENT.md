# Deployment Guide - Biometric Tracking System

Complete deployment guide for the Biometric Tracking System.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LOCAL MACHINE                             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Tracker Service (run_tracker_multi.py)               │    │
│  │  - Camera access (webcam/IP camera)                   │    │
│  │  - Local models (YOLO, InsightFace, OSNet)          │    │
│  │  - Writes to tracker_state.json                       │    │
│  │  - Sends data to Render backend (optional)            │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP/REST API
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RENDER.COM (Cloud)                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────┐  │
│  │  Backend API    │    │  Dashboard     │    │  Redis  │  │
│  │  (FastAPI)     │◀───│  (Streamlit)  │    │  Cache  │  │
│  │  Port: 8000    │    │  Port: 8501    │    │         │  │
│  └─────────────────┘    └─────────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Deployment Options

| Option | Tracker | Backend | Dashboard | Best For |
|---------|---------|---------|-----------|----------|
| **Local + Render** | Local | Render.com | Render.com | Production (recommended) |
| **Fully Cloud** | Cloud VM | Cloud VM | Cloud VM | Development/testing |
| **Edge (Jetson)** | Jetson Nano | Jetson Nano | Remote/Edge | On-premise surveillance |
| **Edge (RPi)** | RPi | RPi | Remote | Budget edge deployment |

---

## Option 1: Local Tracker + Render Cloud (Recommended)

### Why This Architecture?
- **Tracker needs camera access** → runs locally
- **Backend/Dashboard** → deployed to Render.com for remote access
- **Hybrid approach** → best of both worlds

### Step 1: Deploy to Render.com

1. Push code to GitHub:
   ```bash
   git add .
   git commit -m "feat: ready for deployment"
   git push origin main
   ```

2. Go to [Render.com](https://render.com) → **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render auto-detects `render.yaml`
5. Click **"Apply Blueprint"**

### Step 2: Configure Secrets

In GitHub → **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `RENDER_API_KEY` | Get from Render dashboard → Account settings |
| `RENDER_BACKEND_SERVICE_ID` | Backend service ID from Render |
| `RENDER_DASHBOARD_SERVICE_ID` | Dashboard service ID from Render |

### Step 3: Run Tracker Locally

1. Update `config.py` with Render URLs:
   ```python
   BACKEND_URL = "https://biometric-backend.onrender.com"
   DASHBOARD_URL = "https://biometric-dashboard.onrender.com"
   USE_REMOTE_BACKEND = True
   ```

2. Run tracker:
   ```bash
   pip install -r requirements.txt
   python run_tracker_multi.py
   ```

### Step 4: Verify

- **Backend**: https://biometric-backend.onrender.com/
- **Dashboard**: https://biometric-dashboard.onrender.com/
- **Local Tracker**: Check terminal output

---

## Option 2: Fully Cloud Deployment

Deploy everything to a cloud VM (AWS EC2, GCP Compute Engine, Azure VM).

### Quick Deploy

```bash
# SSH into your cloud VM
ssh user@your-vm-ip

# Clone repository
git clone https://github.com/your-username/biometric-tracking-system.git
cd biometric-tracking-system

# Copy environment file
cp .env.example .env
nano .env  # Edit as needed

# Deploy with Docker Compose
docker-compose -f docker-compose.yml \
             -f deploy/cloud/docker-compose.cloud.yml \
             up -d

# Check status
docker-compose ps
```

See [deploy/cloud/README.md](deploy/cloud/README.md) for details.

---

## Option 3: Edge Deployment (Jetson Nano)

For on-premise surveillance with local processing.

### Prerequisites
- NVIDIA Jetson Nano (with JetPack installed)
- CSI camera or USB webcam
- Monitor + keyboard (or SSH setup)

### Deploy

```bash
# On Jetson Nano:
# 1. Flash JetPack to SD card (includes CUDA, Docker)

# 2. Copy project to Jetson:
scp -r biometric-tracking-system user@jetson-ip:~/

# 3. SSH into Jetson:
ssh user@jetson-ip
cd biometric-tracking-system

# 4. Deploy with Jetson-optimized config:
docker-compose -f docker-compose.yml \
             -f deploy/edge/jetson/docker-compose.jetson.yml \
             up -d

# 5. Check status:
docker-compose ps
```

See [deploy/edge/jetson/](deploy/edge/jetson/) for details.

---

## Option 4: Edge Deployment (Raspberry Pi)

Budget-friendly edge deployment.

### Prerequisites
- Raspberry Pi 3+ or 4
- Raspberry Pi OS (64-bit recommended)
- USB webcam
- 16GB+ SD card

### Deploy

```bash
# On Raspberry Pi:
# 1. Flash Raspberry Pi OS to SD card
# 2. Enable camera: sudo raspi-config
# 3. Install Docker:
curl -sSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 4. Copy project:
scp -r biometric-tracking-system pi@rpi-ip:~/
ssh pi@rpi-ip
cd biometric-tracking-system

# 5. Deploy:
docker-compose -f docker-compose.yml \
             -f deploy/edge/rpi/docker-compose.rpi.yml \
             up -d
```

See [deploy/edge/raspberry-pi/](deploy/edge/raspberry-pi/) for details.

---

## CI/CD Pipeline

Automated with GitHub Actions (`.github/workflows/ci-cd.yml`):

| Trigger | Action |
|---------|--------|
| Push to `main` | Run tests → Build Docker images → Deploy to Render |
| Pull Request | Run lint & tests |
| Git Tag (v*) | Create GitHub Release |

### Manual Deployment

```bash
# Deploy to Render (triggers GitHub Action)
git tag v1.0.0
git push origin v1.0.0
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DROIDCAM_IP` | `192.168.29.141` | DroidCam IP address |
| `DROIDCAM_PORT` | `4747` | DroidCam port |
| `BACKEND_URL` | `http://localhost:8000` | Backend API URL |
| `DASHBOARD_URL` | `http://localhost:8501` | Dashboard URL |
| `USE_REMOTE_BACKEND` | `False` | Set `True` for Render deployment |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `ENVIRONMENT` | `development` | `production` for cloud/edge |

---

## Monitoring & Maintenance

### Local Tracker
```bash
# View tracker output
python run_tracker_multi.py

# Check tracker_state.json
cat tracker_state.json
```

### Render.com
- Go to [Render Dashboard](https://dashboard.render.com)
- Click your service → **Logs**
- Monitor health checks: `https://biometric-backend.onrender.com/health`

### Cloud VM
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Resource usage
docker stats
```

---

## Troubleshooting

### Tracker can't connect to backend
- Check `BACKEND_URL` in `config.py`
- Verify backend is running: `curl https://biometric-backend.onrender.com/`
- Check CORS settings in `backend/app.py`

### Dashboard not loading
- Verify Streamlit is running: `curl https://biometric-dashboard.onrender.com/`
- Check browser console for errors
- Verify CORS allows dashboard origin

### Camera not found (local tracker)
- Linux: Check `/dev/video*` exists
- Windows: Verify camera in Device Manager
- Try different camera index: `CAMERA_SOURCES = {0: 1, ...}` (try 0, 1, 2...)

### Out of memory (edge devices)
- Reduce `docker-compose.*.yml` resource limits
- Use lighter models (YOLOv8n instead of larger models)
- Disable gait recognition on RPi (too compute-intensive)

---

## Authors

**Prityanshu Yadav** - Project Lead, Core Development  
**Sarthak Senapati** - Co-Developer, Deployment & Testing

---

## Quick Reference

| Task | Command |
|------|---------|
| Run tracker locally | `python run_tracker_multi.py` |
| Start all (Docker) | `docker-compose up -d` |
| Deploy to Render | Push to `main` branch |
| Build images | `make build` (or `docker-compose build`) |
| View logs | `make logs` (or `docker-compose logs -f`) |
| Stop all | `make down` (or `docker-compose down`) |
