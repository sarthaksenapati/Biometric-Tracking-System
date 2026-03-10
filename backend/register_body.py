# backend/register_body.py

import cv2
import numpy as np
from models.reid_model import ReIDModel
from models.detector import PersonDetector
from utils.embeddings import save_embedding

DROIDCAM_URL = "http://192.168.0.183:4747/video"


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


def register_body(person_id):
    source = select_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[REGISTER] ❌ Could not open camera: {source}")
        return

    detector = PersonDetector()
    reid = ReIDModel()

    print(f"\n[REGISTER] Registering body for: {person_id}")
    print("  - Stand fully in frame (head to toe if possible)")
    print("  - Press 's' multiple times from slightly different positions/angles")
    print("  - Vary distance slightly between captures for robustness")
    print("  - Press ESC to quit\n")

    collected = []
    TARGET_SAMPLES = 15

    while True:
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
            cv2.putText(display,
                        f"Press 's' [{len(collected)}/{TARGET_SAMPLES}]",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not person_found:
            cv2.putText(display, "No person detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show which camera is active
        cam_label = "iPhone (DroidCam)" if source != 0 else "Laptop Webcam"
        cv2.putText(display, f"CAM: {cam_label}", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow(f"Register Body — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and person_found:
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
                    emb = emb / np.linalg.norm(emb)
                    collected.append(emb)
                    print(f"  Sample {len(collected)}/{TARGET_SAMPLES} captured  "
                          f"crop=({x2-x1}x{y2-y1})")
                    break

            if len(collected) >= TARGET_SAMPLES:
                stack = np.stack(collected)   # shape (TARGET_SAMPLES, 512)
                save_embedding(person_id, stack, "body")
                print(f"\n[REGISTER] ✅ Saved {TARGET_SAMPLES} body exemplars for {person_id}")
                print(f"           Shape: {stack.shape} — multi-exemplar matching enabled")
                break

        elif key == ord('s') and not person_found:
            print("  ⚠️  No person detected")

        elif key == 27:
            print("[REGISTER] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_body(person_id)