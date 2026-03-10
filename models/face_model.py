# models/face_model.py

import numpy as np
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[FaceModel] insightface not installed — falling back to DeepFace")


class FaceRecognizer:
    def __init__(self):
        if INSIGHTFACE_AVAILABLE:
            import onnxruntime as ort

            # InsightFace ignores providers= in newer versions — it rebuilds
            # InferenceSession internally with its own logic.
            # Fix: monkey-patch onnxruntime.InferenceSession.__init__ BEFORE
            # InsightFace loads any model so every session gets CUDA forced in.
            _original_session_init = ort.InferenceSession.__init__

            def _force_cuda_init(self_session, model_path, sess_options=None,
                                 providers=None, provider_options=None, **kwargs):
                _original_session_init(
                    self_session, model_path,
                    sess_options=sess_options,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    provider_options=provider_options,
                    **kwargs
                )

            ort.InferenceSession.__init__ = _force_cuda_init

            self.app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            # ctx_id=0 = GPU 0 (RTX 4060)
            self.app.prepare(ctx_id=0, det_size=(320, 320))

            # Restore original session init so other code is unaffected
            ort.InferenceSession.__init__ = _original_session_init

            # Verify which provider was actually loaded
            active = []
            for model in self.app.models.values():
                if hasattr(model, 'session'):
                    active += model.session.get_providers()
            if active:
                providers_used = list(set(active))
                gpu = "CUDAExecutionProvider" in providers_used
                print(f"[FaceModel] ONNX providers: {providers_used} — {'GPU' if gpu else 'CPU only'}")
            else:
                print("[FaceModel] Could not verify ONNX providers")

            self.backend = "insightface"
            print("[FaceModel] Using InsightFace (better at distance)")
        else:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.backend = "deepface"
            print("[FaceModel] Using DeepFace FaceNet (limited at distance)")

    def get_embedding(self, crop_img):
        if crop_img is None or crop_img.size == 0:
            return None

        h, w = crop_img.shape[:2]
        if w < 60 or h < 80:
            return None

        if self.backend == "insightface":
            return self._insightface_embedding(crop_img)
        else:
            return self._deepface_embedding(crop_img)

    def _insightface_embedding(self, img):
        try:
            # InsightFace expects BGR (OpenCV format) — no conversion needed
            faces = self.app.get(img)

            if not faces:
                return None

            # Pick the largest face in the crop
            largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embedding = np.array(largest.embedding, dtype=np.float32)

            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return embedding / norm

        except Exception as e:
            print(f"[FaceModel] InsightFace error: {e}")
            return None

    def _deepface_embedding(self, img):
        try:
            result = self.app.represent(
                img,
                model_name="Facenet",
                enforce_detection=False
            )
            if not result:
                return None
            embedding = np.array(result[0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return None
            return embedding / norm
        except Exception as e:
            print(f"[FaceModel] DeepFace error: {e}")
            return None