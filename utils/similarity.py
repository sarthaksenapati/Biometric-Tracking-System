# utils/similarity.py

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def find_best_match(query_embedding: np.ndarray,
                    db: dict,
                    threshold: float = 0.45) -> tuple[str, float]:
    """
    Match query_embedding against a database of embeddings.
    Handles both:
      - Single embeddings  shape (512,)
      - Multi-exemplar     shape (N, 512)  ← new body registration format
    For multi-exemplar, takes the MAX similarity across all stored samples.
    """
    if query_embedding is None:
        return "Unknown", 0.0

    query = query_embedding.flatten().astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-8)

    best_name  = "Unknown"
    best_score = 0.0

    for person_name, emb in db.items():
        emb = np.array(emb, dtype=np.float32)

        # Handle multi-exemplar stacks (N, dim)
        if emb.ndim == 2:
            scores = []
            for row in emb:
                row = row / (np.linalg.norm(row) + 1e-8)
                scores.append(cosine_similarity(query, row))
            score = max(scores)   # best match across all exemplars
        else:
            emb   = emb / (np.linalg.norm(emb) + 1e-8)
            score = cosine_similarity(query, emb)

        if score > best_score:
            best_score = score
            best_name  = person_name

    if best_score < threshold:
        return "Unknown", best_score

    return best_name, best_score