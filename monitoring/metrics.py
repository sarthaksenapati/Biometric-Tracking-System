import time
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY

# ── Metrics Definitions ──────────────────────────────────────────────

fps_gauge = Gauge(
    "biometric_fps",
    "Frames processed per second per camera",
    ["cam_id"],
)

detections_counter = Counter(
    "biometric_detections_total",
    "Total detections per camera",
    ["cam_id"],
)

identities_counter = Counter(
    "biometric_identities_total",
    "Total successful identifications",
    ["cam_id"],
)

camera_status_gauge = Gauge(
    "biometric_camera_status",
    "Camera status: 1=online, 0=offline",
    ["cam_id"],
)

model_load_histogram = Histogram(
    "biometric_model_load_time_seconds",
    "Time to load models",
    ["model_name"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

matching_latency_histogram = Histogram(
    "biometric_matching_latency_seconds",
    "Time for matcher.identify()",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

frames_processed_counter = Counter(
    "biometric_frames_processed_total",
    "Total frames processed",
    ["cam_id"],
)

# ── Helper Functions ──────────────────────────────────────────────────

def update_fps(cam_id, fps):
    fps_gauge.labels(cam_id=str(cam_id)).set(fps)


def record_detection(cam_id):
    detections_counter.labels(cam_id=str(cam_id)).inc()


def record_identification(cam_id):
    identities_counter.labels(cam_id=str(cam_id)).inc()


def update_camera_status(cam_id, online: bool):
    camera_status_gauge.labels(cam_id=str(cam_id)).set(1 if online else 0)


def observe_model_load(model_name, duration_seconds):
    model_load_histogram.labels(model_name=model_name).observe(duration_seconds)


def observe_matching_latency(duration_seconds):
    matching_latency_histogram.observe(duration_seconds)


def record_frame_processed(cam_id):
    frames_processed_counter.labels(cam_id=str(cam_id)).inc()


def get_metrics():
    """Return Prometheus metrics in text format."""
    return generate_latest(REGISTRY)


class ModelLoadTimer:
    """Context manager to time model loading."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        observe_model_load(self.model_name, duration)
        print(f"[METRICS] {self.model_name} loaded in {duration:.2f}s")
