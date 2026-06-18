# utils/config.py

CAMERA_ID = 0   # webcam (later can be RTSP link)

MODEL_PATH = "yolov8n.pt"

CONF_THRESHOLD = 0.4


# ─────────────────────────────────────────────────────────────────────────────
# Modality fusion weights — SINGLE SOURCE OF TRUTH
#
# Three subsystems intentionally weight the biometric modalities differently
# because they solve different problems. The values were previously duplicated
# (and drifted) across files; they now live here so they can be reasoned about
# and tuned in one place. Changing a number here changes it everywhere.
#
# 1) GALLERY_FUSION_WEIGHTS — core.fusion_engine.FusionEngine
#    Used to identify a live person against the registered gallery. Face is the
#    dominant, most-discriminative signal; gait is near-zero on a frontal camera.
#    A trusted identification additionally REQUIRES a face (see Matcher), so this
#    is the strict, high-precision path.
GALLERY_FUSION_WEIGHTS = {"face": 0.65, "body": 0.35, "gait": 0.01}

# 2) CACHE_FUSION_WEIGHTS — core.multi_tracker.SharedIdentityCache
#    Short-term, cross-camera re-identification where the face is frequently NOT
#    visible (person walking away / side-on). Body, gait and clothing colour
#    therefore carry meaningfully more weight than in the gallery path. This path
#    can identify without a face (by design) but only AFTER an entry was seeded
#    by a face-confident match — see the two-tier trust model in the README.
CACHE_FUSION_WEIGHTS = {"face": 0.50, "body": 0.30, "gait": 0.15, "cloth": 0.05}

# 3) ENROLL_MATCH_WEIGHTS — core.multi_tracker.AutoEnroller
#    Used only to decide whether a new track belongs to an existing (still
#    enrolling) candidate. Gait/clothing are not collected during enrollment, so
#    only face and body participate.
ENROLL_MATCH_WEIGHTS = {"face": 0.65, "body": 0.35}