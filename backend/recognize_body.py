# backend/recognize_body.py

import cv2
from models.detector import PersonDetector
from models.reid_model import ReIDModel
from utils.embeddings import load_all_embeddings
from utils.similarity import find_best_match


def recognize_body():
    cap = cv2.VideoCapture(0)

    detector = PersonDetector()
    reid = ReIDModel()
    db = load_all_embeddings("body")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            body_crop = frame[y1:y2, x1:x2]
            emb = reid.get_embedding(body_crop)

            person, score = find_best_match(emb, db)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{person} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Body ReID", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_body()
