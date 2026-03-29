# backend/register.py

import cv2
import numpy as np
from models.face_model import FaceRecognizer
from models.detector import PersonDetector
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


def register_person(person_id):
    source = select_camera()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[REGISTER] ❌ Could not open camera: {source}")
        return

    face_model = FaceRecognizer()
    detector = PersonDetector()

    print(f"\n[REGISTER] Registering face for: {person_id}")
    print("  - Stand 1-1.5 metres from camera")
    print("  - Face well lit, looking straight at camera")
    print("  - Slightly vary your angle (left, front, right, up, down)")
    print("  - Press 's' to capture samples (need 20)")
    print("  - ESC to quit\n")

    collected = []
    TARGET = 20

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        detections = detector.detect(frame)
        face_found = False

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            emb = face_model.get_embedding(crop)

            if emb is not None:
                face_found = True
                color = (0, 255, 0)
                status = f"Face OK ({crop_w}x{crop_h}) — press 's' [{len(collected)}/{TARGET}]"
            else:
                color = (0, 165, 255)
                status = f"Crop too small ({crop_w}x{crop_h}) — move closer!"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if not face_found and not detections:
            cv2.putText(display, "No person detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show which camera is active
        cam_label = "iPhone (DroidCam)" if source != 0 else "Laptop Webcam"
        cv2.putText(display, f"CAM: {cam_label}", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        cv2.imshow(f"Register Face — {person_id}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if face_found:
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    emb = face_model.get_embedding(crop)
                    if emb is not None:
                        emb = emb / np.linalg.norm(emb)
                        collected.append(emb)
                        print(f"  Sample {len(collected)}/{TARGET} captured  "
                              f"crop=({x2-x1}x{y2-y1})")
                        break

                if len(collected) >= TARGET:
                    stack = np.stack(collected)   # shape (TARGET, 512)
                    save_embedding(person_id, stack, "face")
                    print(f"\n[REGISTER] ✅ Saved {TARGET} face exemplars for {person_id}")
                    print(f"           Shape: {stack.shape} — multi-exemplar matching enabled")
                    break
            else:
                print("  ⚠️  Move closer — crop too small for reliable embedding")

        elif key == 27:
            print("[REGISTER] Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_id = input("Enter Person ID: ")
    register_person(person_id)