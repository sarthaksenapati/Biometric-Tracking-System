# backend/register_gait.py

import cv2
import numpy as np
from models.detector import PersonDetector
from models.gait_model import GaitModel
from utils.embeddings import save_embedding

DROIDCAM_URL = "http://192.168.220.178:4747/video"


def select_camera():
    print("\n[CAMERA] Select registration camera:")
    print("  1 → Laptop webcam")
    print("  2 → iPhone (DroidCam WiFi)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "2":
        print(f"[CAMERA] Using iPhone DroidCam: {DROIDCAM_URL}")
        print("[CAMERA] Make sure DroidCam PC client is CLOSED and iPhone app is open.\n")
        return DROIDCAM_URL
    else:
        print("[CAMERA] Using laptop webcam (index 0)\n")
        return 0


def register_gait(person_id):
    source = select_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[REGISTER] ❌ Could not open camera: {source}")
        return

    detector = PersonDetector()
    gait_model = GaitModel()

    print(f"\n[REGISTER] Registering gait for: {person_id}")
    print("  - Walk naturally back and forth in front of the camera")
    print("  - At least 3-4 full steps needed before saving")
    print("  - Press 's' to save when ready | ESC to quit\n")

    frame_sequence = []
    MIN_FRAMES = 20
    MAX_FRAMES = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]
            if body_crop.size == 0:
                continue

            frame_sequence.append(body_crop)

            if len(frame_sequence) > MAX_FRAMES:
                frame_sequence.pop(0)

            color = (0, 255, 0) if len(frame_sequence) >= MIN_FRAMES else (0, 165, 255)
            status = f"Ready — press 's'! ({len(frame_sequence)} frames)" \
                     if len(frame_sequence) >= MIN_FRAMES \
                     else f"Keep walking... ({len(frame_sequence)}/{MIN_FRAMES})"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, status,
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Show which camera is active
        cam_label = "iPhone (DroidCam)" if source != 0 else "Laptop Webcam"
        cv2.putText(display, f"CAM: {cam_label}", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow(f"Register Gait — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(frame_sequence) >= MIN_FRAMES:
                try:
                    if hasattr(gait_model, 'get_embedding'):
                        emb = gait_model.get_embedding(frame_sequence[-MAX_FRAMES:])
                    elif hasattr(gait_model, 'extract_gait_embedding'):
                        emb = gait_model.extract_gait_embedding(frame_sequence[-MAX_FRAMES:])
                    else:
                        print("  ❌ GaitModel has no get_embedding or extract_gait_embedding method")
                        break

                    if emb is not None:
                        save_embedding(person_id, emb, "gait")
                        print(f"\n[REGISTER] ✅ Saved gait embedding for {person_id}")
                        break
                    else:
                        print("  ⚠️  Gait model returned None — try again")
                except Exception as e:
                    print(f"  ❌ Gait extraction failed: {e}")
            else:
                print(f"  ⚠️  Not enough frames yet ({len(frame_sequence)}/{MIN_FRAMES}) — keep walking")

        elif key == 27:
            print("[REGISTER] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_gait(person_id)