# backend/recognize_gait.py

import cv2
from models.detector import PersonDetector
from models.gait_model import GaitModel
from utils.embeddings import load_all_embeddings
from utils.similarity import find_best_match


def recognize_gait():
    cap = cv2.VideoCapture(0)

    detector = PersonDetector()
    gait_model = GaitModel()
    db = load_all_embeddings("gait")

    frame_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            body_crop = frame[y1:y2, x1:x2]

            frame_sequence.append(body_crop)

            if len(frame_sequence) > 20:
                emb = gait_model.extract_gait_embedding(frame_sequence[-20:])
                person, score = find_best_match(emb, db)

                cv2.putText(
                    frame, f"{person} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Gait Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_gait()
