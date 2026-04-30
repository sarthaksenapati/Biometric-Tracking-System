# backend/app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os

app = FastAPI(title="Biometric Tracking System API", version="1.0.0")

# ── CORS Configuration ──────────────────────────────────────────────
# Allow frontend (Streamlit dashboard) to communicate with backend
# Update origins when deploying to production

allowed_origins = [
    "http://localhost:8501",  # Local Streamlit
    "http://localhost:8501",  # Local dashboard
    "https://biometric-dashboard.onrender.com",  # Production dashboard (update with your URL)
]

# Add dashboard URL from environment if set
dashboard_url = os.getenv("DASHBOARD_URL")
if dashboard_url:
    allowed_origins.append(dashboard_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health Check ────────────────────────────────────────────────────


@app.get("/")
def home():
    return {"message": "Biometric Tracking System Running"}


@app.get("/health")
def health_check():
    """Health check endpoint for Docker/Render health checks."""
    return {"status": "healthy", "service": "backend"}


@app.get("/api/status")
def get_status():
    """Get system status."""
    return {
        "backend": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "persons": "/api/persons",
            "cameras": "/api/cameras",
            "events": "/api/events",
        },
    }


# ── API Endpoints (Stubs for future implementation) ──────────────────────


class PersonResponse(BaseModel):
    name: str
    location: Optional[str] = None
    last_seen: Optional[float] = None
    confidence: Optional[float] = None


@app.get("/api/persons", response_model=list[Dict[str, Any]])
def list_persons():
    """List all known persons from embeddings_db/."""
    try:
        persons = []
        if os.path.exists("embeddings_db"):
            files = os.listdir("embeddings_db")
            seen = set()
            for f in files:
                if f.endswith(".npy"):
                    name = f.split("_")[0]
                    if name not in seen:
                        seen.add(name)
                        persons.append({"name": name, "has_embeddings": True})
        return persons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras")
def get_cameras():
    """Get camera status."""
    return {"cameras": [], "message": "Camera data managed by local tracker"}


@app.get("/api/events")
def get_events(limit: int = 50):
    """Get recent events from tracker_state.json."""
    try:
        if os.path.exists("tracker_state.json"):
            with open("tracker_state.json") as f:
                data = json.load(f)
            events = data.get("events", [])[-limit:]
            return {"events": events}
        return {"events": [], "message": "No tracker state available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Tracking Data Endpoint (for local tracker → remote backend) ─────────


class TrackingData(BaseModel):
    active_people: list
    events: list
    cameras: list


@app.post("/api/tracker/update")
def update_tracker_data(data: TrackingData):
    """Receive tracking data from local tracker and store for dashboard."""
    try:
        with open("tracker_state.json", "w") as f:
            json.dump(data.model_dump(), f, indent=2)
        return {"status": "success", "message": "Tracker data updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
