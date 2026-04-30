# backend/register_gait.py
#
# Registers gait embeddings for a person.
# Collects multiple walking sequences and averages them for robustness.
#
# USAGE:
#   python backend/register_gait.py

import cv2
import numpy as np
from models.detector import PersonDetector
from models.gait_model import GaitModel
from utils.embeddings import save_embedding
from config import DROIDCAM_IP, DROIDCAM_PORT

DROIDCAM_URL = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
MIN_FRAMES = 45  # ~1.5s at 30fps — enough for one full gait cycle
MAX_FRAMES = 60  # keep a rolling window of 2s
TARGET_SEQUENCES = 3  # collect 3 separate walks and average them


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


def register_gait(person_id: str):
    source = select_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[REGISTER] ❌ Could not open camera: {source}")
        return

    detector = PersonDetector()
    gait_model = GaitModel()

    print(f"\n[REGISTER] Registering gait for: {person_id}")
    print(f"  Sequences needed : {TARGET_SEQUENCES}")
    print(f"  Frames per seq   : {MIN_FRAMES}–{MAX_FRAMES}  (~1.5–2s of walking)")
    print("  Walk naturally past the camera — one full pass per sequence")
    print("  Press 's' to save each sequence when ready | ESC to cancel\n")

    sequences_collected = []  # list of embeddings, one per walk
    frame_sequence = []  # rolling buffer of body crops

    while len(sequences_collected) < TARGET_SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)
        seq_num = len(sequences_collected) + 1

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]
            if body_crop.size == 0:
                continue

            # Rolling window
            frame_sequence.append(body_crop)
            if len(frame_sequence) > MAX_FRAMES:
                frame_sequence.pop(0)

            ready = len(frame_sequence) >= MIN_FRAMES
            color = (0, 255, 0) if ready else (0, 165, 255)
            status = (
                f"Ready — press 's'! ({len(frame_sequence)} frames)"
                if ready
                else f"Keep walking... ({len(frame_sequence)}/{MIN_FRAMES})"
            )

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, status, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # ── HUD ──────────────────────────────────────────────────────────────
        cam_label = "DroidCam" if source != 0 else "Laptop Webcam"

        cv2.rectangle(display, (0, 0), (display.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"WALK {seq_num}/{TARGET_SEQUENCES} — walk naturally past the camera",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )

        # Progress bar for sequences
        bar_x, bar_y, bar_w, bar_h_size = 10, 45, 300, 12
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_size), (60, 60, 60), -1)
        filled = int(bar_w * (len(sequences_collected) / TARGET_SEQUENCES))
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h_size), (0, 200, 100), -1)
        cv2.putText(
            display,
            f"Sequences: {len(sequences_collected)}/{TARGET_SEQUENCES}",
            (bar_x + bar_w + 10, bar_y + bar_h_size),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        cv2.putText(
            display, f"CAM: {cam_label}", (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1
        )

        cv2.imshow(f"Register Gait — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(frame_sequence) < MIN_FRAMES:
                print(f"  ⚠️  Not enough frames yet " f"({len(frame_sequence)}/{MIN_FRAMES}) — keep walking")
                continue

            try:
                # Extract embedding from current rolling window
                if hasattr(gait_model, "get_embedding"):
                    emb = gait_model.get_embedding(frame_sequence[-MAX_FRAMES:])
                elif hasattr(gait_model, "extract_gait_embedding"):
                    emb = gait_model.extract_gait_embedding(frame_sequence[-MAX_FRAMES:])
                else:
                    print("  ❌ GaitModel has no get_embedding or " "extract_gait_embedding method")
                    break

                if emb is not None:
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    sequences_collected.append(emb)
                    print(
                        f"  ✅ Sequence {len(sequences_collected)}/{TARGET_SEQUENCES} "
                        f"saved  ({len(frame_sequence)} frames used)"
                    )

                    # Reset buffer for next walk
                    frame_sequence.clear()

                    if len(sequences_collected) < TARGET_SEQUENCES:
                        print(f"     ➡️  Walk {len(sequences_collected) + 1} — " f"walk past again and press 's'\n")
                else:
                    print("  ⚠️  Gait model returned None — try again")

            except Exception as e:
                print(f"  ❌ Gait extraction failed: {e}")

        elif key == 27:
            print("[REGISTER] Cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return

    # ── Average all sequences and save ───────────────────────────────────────
    if len(sequences_collected) >= TARGET_SEQUENCES:
        stacked = np.stack(sequences_collected)  # (TARGET_SEQUENCES, dim)
        avg_emb = np.mean(stacked, axis=0)  # (dim,)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

        save_embedding(person_id, avg_emb, "gait")
        print(f"\n[REGISTER] ✅ Saved averaged gait embedding for '{person_id}'")
        print(f"           Sequences averaged : {TARGET_SEQUENCES}")
        print(f"           Embedding shape    : {avg_emb.shape}")
        print(f"           Frames per seq     : {MIN_FRAMES}–{MAX_FRAMES}")
    else:
        print(
            f"\n[REGISTER] ⚠️  Only {len(sequences_collected)} sequences collected "
            f"— not saved. Need {TARGET_SEQUENCES}."
        )

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    person_id = input("Enter Person ID: ").strip()
    if person_id:
        register_gait(person_id)
    else:
        print("❌ Person ID cannot be empty.")
