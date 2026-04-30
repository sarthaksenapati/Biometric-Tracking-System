# backend/recognize.py

import cv2
from models.face_model import FaceRecognizer
from utils.embeddings import load_all_embeddings
from utils.similarity import find_best_match
from models.detector import PersonDetector


def recognize_live():
    cap = cv2.VideoCapture(0)

    detector = PersonDetector()
    face_model = FaceRecognizer()
    db = load_all_embeddings("face")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Safe crop
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]

            emb = face_model.get_embedding(face_crop)
            if emb is None:
                continue

            person, score = find_best_match(emb, db)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{person} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_live()
