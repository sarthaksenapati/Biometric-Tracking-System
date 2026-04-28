# backend/app.py

import os
import json
import time
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Instrument the app with Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ── Health Check Endpoints ──────────────────────────────────────

@app.get("/health/live")
def health_live():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health/ready")
def health_ready():
    checks = {}
    overall = "healthy"

    # Check database
    try:
        from database.db import get_connection
        conn = get_connection()
        start = time.time()
        conn.close()
        checks["database"] = {"status": "ok", "latency_ms": round((time.time() - start) * 1000, 1)}
    except Exception as e:
        checks["database"] = {"status": "error", "error": str(e)}
        overall = "unhealthy"

    # Check Redis
    try:
        from cache.redis_cache import get_redis_client
        client = get_redis_client()
        start = time.time()
        client.ping()
        client.close()
        checks["redis"] = {"status": "ok", "latency_ms": round((time.time() - start) * 1000, 1)}
    except Exception as e:
        checks["redis"] = {"status": "error", "error": str(e)}
        if overall == "healthy":
            overall = "degraded"

    return {"status": overall, "timestamp": time.time(), "checks": checks}


@app.get("/health/cameras")
def health_cameras():
    cameras = {}
    try:
        if os.path.exists("tracker_state.json"):
            with open("tracker_state.json", "r") as f:
                state = json.load(f)
            for cam in state.get("cameras", []):
                cameras[str(cam["cam_id"])] = {
                    "status": "online" if cam.get("online") else "offline",
                    "location": cam.get("location", ""),
                }
    except Exception as e:
        return {"status": "error", "error": str(e)}

    online = sum(1 for c in cameras.values() if c["status"] == "online")
    return {
        "status": "healthy" if online > 0 else "degraded",
        "cameras": cameras,
        "online_count": online,
        "total": len(cameras),
    }


@app.get("/health/models")
def health_models():
    models = {}
    try:
        from database.db import DB_AVAILABLE
        models["database"] = "loaded" if DB_AVAILABLE else "unavailable"
    except Exception:
        models["database"] = "error"

    try:
        from cache.redis_cache import REDIS_AVAILABLE
        models["redis_cache"] = "loaded" if REDIS_AVAILABLE else "unavailable"
    except Exception:
        models["redis_cache"] = "error"

    models["detector"] = "not_loaded"
    models["face_model"] = "not_loaded"
    models["reid_model"] = "not_loaded"
    models["gait_model"] = "not_loaded"

    return {"status": "healthy", "models": models}


@app.get("/health/full")
def health_full():
    live = health_live()
    ready = health_ready()
    cams = health_cameras()
    mods = health_models()

    checks = ready["checks"].copy()
    checks["cameras"] = cams["cameras"]
    checks["models"] = mods["models"]

    # Count recent events from tracker state
    try:
        if os.path.exists("tracker_state.json"):
            with open("tracker_state.json", "r") as f:
                state = json.load(f)
            checks["recent_events"] = len(state.get("events", []))
            checks["active_people"] = len(state.get("active_people", []))
    except Exception:
        pass

    status = "healthy"
    if ready["status"] != "healthy" or cams["status"] != "healthy":
        status = "degraded"
    if ready["status"] == "unhealthy":
        status = "unhealthy"

    return {"status": status, "timestamp": time.time(), "checks": checks}


# ── Alerts Endpoint ─────────────────────────────────────────

@app.get("/alerts")
def get_alerts():
    try:
        from monitoring.alerts import get_alert_manager
        manager = get_alert_manager()
        return {"alerts": manager.get_active_alerts(), "total": len(manager.get_active_alerts())}
    except Exception as e:
        return {"alerts": [], "error": str(e)}


# ── Prometheus Metrics ─────────────────────────────────────────

@app.get("/metrics")
def metrics():
    from monitoring.metrics import get_metrics
    return PlainTextResponse(get_metrics(), media_type="text/plain")


# ── Root ──────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Biometric Tracking System Running"}
