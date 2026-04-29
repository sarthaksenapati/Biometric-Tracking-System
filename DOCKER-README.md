# Docker Setup - Biometric Tracking System

This directory contains Docker configuration for containerizing the biometric tracking system.

**Authors:** Prityanshu Yadav, Sarthak Senapati

## Services

| Service | Description | Port |
|----------|-------------|------|
| **tracker** | Multi-camera tracker with OpenCV | 8502 |
| **backend** | FastAPI backend API | 8000 |
| **dashboard** | Streamlit web dashboard | 8501 |
| **redis** | Redis for caching and messaging | 6379 |

## Quick Start

### Prerequisites
- Docker Desktop installed
- Docker Compose v2+

### Linux (with webcam access)

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your camera configuration
nano .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Windows (IP cameras only)

On Windows, local webcam access is not available in Docker containers. Use IP cameras (DroidCam, RTSP streams).

```bash
# Copy environment file
copy .env.example .env

# Edit .env with your DroidCam IP
# DROIDCAM_IP=your_phone_ip
# DROIDCAM_PORT=4747

# Start services (excluding tracker - run tracker on host)
docker-compose -f docker-compose.yml -f docker-compose.windows.yml up -d

# Or run tracker on host and only containerize backend+dashboard:
# Terminal 1 (host): python run_tracker_multi.py
# Terminal 2: docker-compose -f docker-compose.yml -f docker-compose.windows.yml up -d backend dashboard
```

## Running Tracker on Host (Windows/Mac)

For local webcam access, run the tracker directly on the host:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tracker
python run_tracker_multi.py

# In another terminal, start backend and dashboard in Docker
docker-compose -f docker-compose.yml -f docker-compose.windows.yml up -d backend dashboard
```

## Building Individual Services

```bash
# Build tracker
docker build -f Dockerfile.tracker -t biometric-tracker .

# Build backend
docker build -f Dockerfile.backend -t biometric-backend .

# Build dashboard
docker build -f Dockerfile.dashboard -t biometric-dashboard .
```

## Volumes and Data Persistence

The following data is persisted through Docker volumes or bind mounts:

- `embeddings_db/` - Face/body/gait embeddings
- `tracker_state.json` - Current tracking state
- `tracker_history.json` - Historical tracking data
- `unknown_persons.json` - Auto-enrollment data

## Camera Access Notes

### Linux
- Webcam: `--device /dev/video0` (included in docker-compose.yml)
- IP Camera: Configure in `config.py` or use DroidCam

### Windows
- Webcam: Not accessible in Docker - run tracker on host
- IP Camera: Works via DroidCam, RTSP, or HTTP streams
- DroidCam: Set `DROIDCAM_IP` and `DROIDCAM_PORT` in `.env`

### Mac
- Webcam: Limited Docker access - run tracker on host
- IP Camera: Works via network streams

## Troubleshooting

### OpenCV window not showing in Docker
The tracker uses Xvfb (virtual framebuffer) in Docker. The OpenCV window won't be visible. Use the Streamlit dashboard to monitor detections.

### Camera not found
- Linux: Check `/dev/video*` exists
- Windows/Mac: Run tracker on host or use IP cameras

### Redis connection refused
```bash
docker-compose restart redis
```

### Rebuild after code changes
```bash
docker-compose up -d --build
```

## Development Mode

For development with hot-reload:

```bash
# Uncomment volume mounts in docker-compose.yml to mount source code
# Then restart with:
docker-compose up -d
```

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v

# Remove all related images
docker rmi biometric-tracker biometric-backend biometric-dashboard
```
