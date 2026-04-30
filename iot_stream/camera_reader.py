# iot_stream/camera_reader.py

import cv2
from models.detector import PersonDetector
from utils.config import CAMERA_ID


def run_camera():
    cap = cv2.VideoCapture(CAMERA_ID)

    detector = PersonDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
