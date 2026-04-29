# run_tracker_multi.py
#
# Runs the multi-camera tracker with OpenCV live video window.
# Writes tracker_state.json every second so the Streamlit dashboard
# can read it without needing to load models or open cameras itself.
#
# USAGE:
#   Terminal 1:  python run_tracker_multi.py
#   Terminal 2:  python -m streamlit run dashboard.py
#
# CAMERA CONFIG:
#   Edit config.py in the project root to change IP, sources, or locations.
#   You never need to touch this file for camera changes.

import os
import time
import json
import threading

# Initialize database if using PostgreSQL
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"
if USE_DATABASE:
    try:
        from db.connection import init_db
        init_db()
        print("[MAIN] ✅ Database initialized")
    except Exception as e:
        print(f"[MAIN] ⚠️  Database init failed: {e}")
        print("[MAIN] Continuing with file-based storage...")

from core.multi_tracker import MultiCameraTracker
from config import CAMERA_SOURCES, CAMERA_LOCATIONS   # ← all camera config lives here

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_WRITE_INTERVAL = 1          # seconds between dashboard state writes
STATE_FILE           = "tracker_state.json"

# ─────────────────────────────────────────────────────────────────────────────
# State writer — background thread, writes JSON for Streamlit dashboard
# ─────────────────────────────────────────────────────────────────────────────

def state_writer(tracker: MultiCameraTracker, stop_event: threading.Event):
    """
    Every STATE_WRITE_INTERVAL seconds, dump structured tracker data
    to tracker_state.json so the Streamlit dashboard can read it.
    """
    while not stop_event.is_set():
        try:
            data = tracker.get_structured_data()
            data["written_at"]    = time.time()
            data["retention_min"] = tracker.identity_manager.RETENTION_SECONDS // 60

            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[STATE] ⚠️  Could not write state file: {e}")

        stop_event.wait(STATE_WRITE_INTERVAL)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Print the active camera config so you can verify on startup
    print("\n[CONFIG] Camera sources loaded from config.py:")
    for cam_id, source in CAMERA_SOURCES.items():
        location = CAMERA_LOCATIONS.get(cam_id, f"Cam{cam_id}")
        print(f"         Cam{cam_id} → '{location}'  (source={source})")
    print()

    tracker = MultiCameraTracker(
        cam_sources   = CAMERA_SOURCES,
        cam_locations = CAMERA_LOCATIONS,
    )

    # Start background state writer thread
    stop_event    = threading.Event()
    writer_thread = threading.Thread(
        target=state_writer,
        args=(tracker, stop_event),
        name="StateWriterThread",
        daemon=True,
    )
    writer_thread.start()
    print(f"[STATE] 📝 Writing dashboard state to '{STATE_FILE}' "
          f"every {STATE_WRITE_INTERVAL}s\n")

    try:
        tracker.run()   # blocking — shows OpenCV window, press ESC to quit
    finally:
        stop_event.set()
        writer_thread.join(timeout=3)
        print("[STATE] State writer stopped.")