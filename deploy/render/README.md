# Render.com Deployment Guide

Deploy the Backend API and Dashboard to Render.com, while running the Tracker locally.

## Architecture

```
┌─────────────────────┐      ┌──────────────────────┐
│  Local Machine      │      │  Render.com (Cloud)  │
│                     │      │                      │
│  Tracker Service    ├─────▶  Backend API         │
│  (run_tracker)     │      │  (FastAPI)           │
│                     │      │                      │
│  - Camera access   │      │  Dashboard           │
│  - Local models    │      │  (Streamlit)         │
│                     │      │                      │
└─────────────────────┘      └──────────────────────┘
         │                           │
         └───────────────────────────┘
              Communicate via HTTP/REST
```

## Step 1: Prepare Your Repository

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "feat: ready for Render deployment"
   git push origin main
   ```

2. Ensure `render.yaml` is in the root directory

## Step 2: Connect to Render

1. Go to [Render.com](https://render.com) and sign up/log in
2. Click **"New +"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will detect `render.yaml` automatically
5. Click **"Apply Blueprint"**

## Step 3: Configure Environment Variables

In Render dashboard, add these to both backend and dashboard services:

| Key | Value | Description |
|-----|-------|-------------|
| `PYTHON_VERSION` | `3.10.0` | Python version |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `ENVIRONMENT` | `production` | Environment mode |

## Step 4: Set Up Local Tracker

Since the tracker needs camera access, it runs locally:

1. **Update `config.py`** to point to Render backend:
   ```python
   # config.py (local machine)
   BACKEND_URL = "https://biometric-backend.onrender.com"
   DASHBOARD_URL = "https://biometric-dashboard.onrender.com"
   ```

2. **Run tracker locally:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run tracker (it will connect to Render backend)
   python run_tracker_multi.py
   ```

3. **Optional: Use local API endpoint**
   - If you want the tracker to also write to local files (for dashboard):
   - Keep `tracker_state.json` writing enabled
   - Dashboard on Render can't read local files, so use the API instead

## Step 5: Verify Deployment

### Backend API:
```bash
curl https://biometric-backend.onrender.com/
# Expected: {"message": "Biometric Tracking System Running"}
```

### Dashboard:
Open in browser: `https://biometric-dashboard.onrender.com`

### Local Tracker:
Check terminal output for successful connection to backend.

## CI/CD with GitHub Actions

The `.github/workflows/ci-cd.yml` automates:
1. **Lint & Test** (on every push/PR)
2. **Build Docker images** (on push to main)
3. **Deploy to Render** (on push to main)

### Required GitHub Secrets

Go to **GitHub repo → Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `RENDER_API_KEY` | Your Render API key (get from Render dashboard) |
| `RENDER_BACKEND_SERVICE_ID` | Backend service ID from Render |
| `RENDER_DASHBOARD_SERVICE_ID` | Dashboard service ID from Render |

## Monitoring

### View Render Logs:
- Go to Render dashboard → Your service → Logs

### Check Health:
```bash
curl https://biometric-backend.onrender.com/health
```

## Troubleshooting

### Backend not connecting to Redis:
- Check Redis service is running in Render
- Verify `REDIS_URL` environment variable

### Dashboard can't reach backend:
- Ensure CORS is enabled in backend (`backend/app.py`)
- Check backend URL is correct

### Local tracker not working:
- Verify camera access: `python test_cam.py`
- Check `config.py` has correct backend URL

## Free Tier Limitations

Render free tier:
- Services spin down after 15 minutes of inactivity
- First request after spin-up takes ~30-60 seconds
- Limited to 750 hours/month

For production, upgrade to Starter plan ($7/month per service).
