# backend/register_body.py
#
# Registers body (ReID) embeddings for a person.
# Captures samples from multiple angles for robust real-world matching.
#
# USAGE:
#   python backend/register_body.py

import cv2
import numpy as np
from models.reid_model import ReIDModel
from models.detector import PersonDetector
from utils.embeddings import save_embedding
from config import DROIDCAM_IP, DROIDCAM_PORT

DROIDCAM_URL = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
SAMPLES_PER_ANGLE = 4  # samples captured at each angle
CAPTURE_ANGLES = [
    "FRONT — face the camera directly",
    "LEFT SIDE — turn your left shoulder toward camera",
    "RIGHT SIDE — turn your right shoulder toward camera",
    "FRONT FAR — step back ~1.5m, face camera",
]
TARGET_TOTAL = SAMPLES_PER_ANGLE * len(CAPTURE_ANGLES)  # 16 samples total


# ─────────────────────────────────────────────────────────────────────────────
# Camera selection
# ─────────────────────────────────────────────────────────────────────────────


def select_camera():
    print("\n[CAMERA] Select registration camera:")
    print("  1 → Laptop webcam")
    print("  2 → DroidCam (WiFi)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        print(f"[CAMERA] Using DroidCam: {DROIDCAM_URL}")
        print("[CAMERA] Make sure DroidCam PC client is CLOSED and phone app is open.\n")
        return DROIDCAM_URL
    print("[CAMERA] Using laptop webcam (index 0)\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────


def register_body(person_id: str):
    source = select_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[REGISTER] ❌ Could not open camera: {source}")
        return

    detector = PersonDetector()
    reid = ReIDModel()

    print(f"\n[REGISTER] Registering body for: {person_id}")
    print(f"  Total samples needed : {TARGET_TOTAL} ({SAMPLES_PER_ANGLE} per angle)")
    print(f"  Angles               : {len(CAPTURE_ANGLES)}")
    print("  Press 's' to capture a sample | ESC to cancel\n")

    collected = []
    angle_idx = 0
    angle_counts = [0] * len(CAPTURE_ANGLES)

    while angle_idx < len(CAPTURE_ANGLES):
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)
        person_found = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]
            if body_crop.size == 0:
                continue

            person_found = True
            color = (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            samples_this_angle = angle_counts[angle_idx]
            cv2.putText(
                display,
                f"Press 's' [{samples_this_angle}/{SAMPLES_PER_ANGLE}]",
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        if not person_found:
            cv2.putText(
                display,
                "No person detected — step into frame",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        # ── HUD ──────────────────────────────────────────────────────────────
        angle_instruction = CAPTURE_ANGLES[angle_idx]
        cam_label = "DroidCam" if source != 0 else "Laptop Webcam"
        total_so_far = sum(angle_counts)

        # Instruction banner at top
        cv2.rectangle(display, (0, 0), (display.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"ANGLE {angle_idx + 1}/{len(CAPTURE_ANGLES)}: {angle_instruction}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
        cv2.putText(
            display,
            f"Total collected: {total_so_far}/{TARGET_TOTAL}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

        # Camera label at bottom
        cv2.putText(
            display, f"CAM: {cam_label}", (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1
        )

        cv2.imshow(f"Register Body — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if not person_found:
                print("  ⚠️  No person detected — step into frame first")
                continue

            # Capture from first detected person
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                body_crop = frame[y1:y2, x1:x2]
                if body_crop.size == 0:
                    continue

                emb = reid.get_embedding(body_crop)
                if emb is not None:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    collected.append(emb)
                    angle_counts[angle_idx] += 1
                    print(
                        f"  ✅ Angle {angle_idx + 1} sample "
                        f"{angle_counts[angle_idx]}/{SAMPLES_PER_ANGLE} captured  "
                        f"crop=({x2-x1}x{y2-y1})"
                    )
                    break

            # Move to next angle when current is complete
            if angle_counts[angle_idx] >= SAMPLES_PER_ANGLE:
                angle_idx += 1
                if angle_idx < len(CAPTURE_ANGLES):
                    next_angle = CAPTURE_ANGLES[angle_idx]
                    print(f"\n  ➡️  NEXT ANGLE: {next_angle}\n")

        elif key == 27:
            print("[REGISTER] Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return

    # ── Save ─────────────────────────────────────────────────────────────────
    if len(collected) >= TARGET_TOTAL:
        stack = np.stack(collected)  # shape: (TARGET_TOTAL, embedding_dim)
        save_embedding(person_id, stack, "body")
        print(f"\n[REGISTER] ✅ Saved {TARGET_TOTAL} body samples for '{person_id}'")
        print(f"           Shape: {stack.shape}")
        print(f"           Angles covered: {len(CAPTURE_ANGLES)} " f"({SAMPLES_PER_ANGLE} samples each)")
        print(f"           Multi-exemplar matching enabled ✓")
    else:
        print(f"\n[REGISTER] ⚠️  Only {len(collected)} samples collected — not saved.")

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    person_id = input("Enter Person ID: ").strip()
    if person_id:
        register_body(person_id)
    else:
        print("❌ Person ID cannot be empty.")
