import threading
from prometheus_client import start_http_server
from core.tracker import LiveTracker

# Start Prometheus metrics HTTP server for scraping
METRICS_PORT = 8001
start_http_server(METRICS_PORT)
print(f"[METRICS] Prometheus metrics exposed on port {METRICS_PORT}")

tracker = LiveTracker(cam_id=0, source=0)
tracker.run()
