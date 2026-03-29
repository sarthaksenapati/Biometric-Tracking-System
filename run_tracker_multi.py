# run_tracker_multi.py
#
# Runs the multi-camera tracker with OpenCV live video window.
# Writes tracker_state.json every second so the Streamlit dashboard
# can read it without needing to load models or open cameras itself.
#
# USAGE:
#   Terminal 1:  python run_tracker_multi.py     ← live video + tracking
#   Terminal 2:  python -m streamlit run dashboard.py  ← monitoring dashboard

import time
import json
import threading
from core.multi_tracker import MultiCameraTracker

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit to match your setup
# ─────────────────────────────────────────────────────────────────────────────

sources = {
    0: 0,
    # 1: "http://192.168.220.178:4747/video",
}

cam_locations = {
    0: "Main Entrance",
    1: "Library Gate",
}

STATE_FILE        = "tracker_state.json"
STATE_WRITE_INTERVAL = 1.0   # seconds between state file writes

# ─────────────────────────────────────────────────────────────────────────────
# State writer — runs in a background thread, writes JSON for dashboard
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
    tracker = MultiCameraTracker(
        cam_sources   = sources,
        cam_locations = cam_locations,
    )

    # Start background state writer
    stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=state_writer,
        args=(tracker, stop_event),
        name="StateWriterThread",
        daemon=True,
    )
    writer_thread.start()
    print(f"[STATE] 📝 Writing dashboard state to '{STATE_FILE}' every {STATE_WRITE_INTERVAL}s\n")

    try:
        tracker.run()   # blocking — shows OpenCV window, press ESC to quit
    finally:
        stop_event.set()
        writer_thread.join(timeout=3)
        print("[STATE] State writer stopped.")