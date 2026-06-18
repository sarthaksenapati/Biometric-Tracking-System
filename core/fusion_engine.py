import numpy as np

from utils.config import GALLERY_FUSION_WEIGHTS


class FusionEngine:
    def __init__(self):
        # Weights are centralized in utils/config.py (GALLERY_FUSION_WEIGHTS) as
        # the single source of truth. Rationale:
        #   face:  0.65  — primary signal, high discriminability
        #   body:  0.35  — strong secondary signal (OSNet MSMT17 when loaded)
        #   gait:  0.01  — frontal camera limits gait utility, near-zero weight
        # A copy is taken so per-instance tweaks never mutate the shared dict.
        self.default_weights = dict(GALLERY_FUSION_WEIGHTS)

    def compute_final_score(
        self,
        face_score=None,
        body_score=None,
        gait_score=None,
        attr_score=None,
        verbose=False
    ):
        scores  = []
        weights = []
        labels  = []

        if face_score is not None:
            # Clip to 0 — a negative face score means the face was unrecognisable
            # (bad angle, motion blur, occlusion). Letting it go negative actively
            # drags down the final score even when body is strongly matched.
            face_score = max(0.0, face_score)
            scores.append(face_score)
            weights.append(self.default_weights["face"])
            labels.append(f"face={face_score:.3f}")

        if body_score is not None:
            scores.append(body_score)
            weights.append(self.default_weights["body"])
            labels.append(f"body={body_score:.3f}")

        if gait_score is not None:
            scores.append(gait_score)
            weights.append(self.default_weights["gait"])
            labels.append(f"gait={gait_score:.3f}")

        if len(scores) == 0:
            return 0.0, False

        scores  = np.array(scores)
        weights = np.array(weights)

        # Normalize so available modalities always sum to 1
        weights = weights / np.sum(weights)

        final_score = float(np.sum(scores * weights))

        # Only trust the result if face was available.
        # Body + gait alone cannot distinguish these subjects reliably
        # at close range with a frontal camera.
        trusted = face_score is not None

        if verbose:
            trust_str = "trusted" if trusted else "UNTRUSTED - no face"
            print(f"  [Fusion] {' | '.join(labels)} → final={final_score:.3f} ({trust_str})")

        return final_score, trusted