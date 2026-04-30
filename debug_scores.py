"""
Run this ONCE to see what scores each person's stored embeddings
give against each other. This tells you what threshold to actually use.

Usage:
    python debug_scores.py
"""

import os
import numpy as np


def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(a / na, b / nb))


DB_PATH = "embeddings_db"

# Load all embeddings
database = {}
for file in os.listdir(DB_PATH):
    if not file.endswith(".npy"):
        continue
    parts = file.replace(".npy", "").split("_")
    if len(parts) < 2:
        continue
    name, modality = parts[0], parts[1].lower()
    if modality not in ("face", "body", "gait"):
        continue
    emb = np.load(os.path.join(DB_PATH, file))
    if name not in database:
        database[name] = {}
    database[name][modality] = emb

people = list(database.keys())
modalities = ["face", "body", "gait"]

print("=" * 60)
print("SELF-SIMILARITY (how well each person matches themselves)")
print("=" * 60)
for person in people:
    for mod in modalities:
        emb = database[person].get(mod)
        if emb is not None:
            sim = cosine_similarity(emb, emb)
            print(f"  {person:15s} {mod:5s} self-sim = {sim:.4f}")

print()
print("=" * 60)
print("CROSS-SIMILARITY (how much they look like each other)")
print("This is what causes false matches!")
print("=" * 60)
for i, p1 in enumerate(people):
    for p2 in people[i + 1 :]:
        for mod in modalities:
            e1 = database[p1].get(mod)
            e2 = database[p2].get(mod)
            sim = cosine_similarity(e1, e2)
            if sim is not None:
                flag = " ⚠️  HIGH CROSS-SIM" if sim > 0.5 else ""
                print(f"  {p1:15s} vs {p2:15s} | {mod:5s} = {sim:.4f}{flag}")

print()
print("=" * 60)
print("RECOMMENDED THRESHOLD")
print("=" * 60)

# Find max cross-sim and min self-sim per modality
for mod in modalities:
    self_sims = []
    cross_sims = []
    for person in people:
        emb = database[person].get(mod)
        if emb is not None:
            self_sims.append(cosine_similarity(emb, emb))
    for i, p1 in enumerate(people):
        for p2 in people[i + 1 :]:
            e1, e2 = database[p1].get(mod), database[p2].get(mod)
            s = cosine_similarity(e1, e2)
            if s is not None:
                cross_sims.append(s)

    if self_sims and cross_sims:
        print(f"  {mod:5s} | min self-sim={min(self_sims):.3f} | max cross-sim={max(cross_sims):.3f}")
        midpoint = (min(self_sims) + max(cross_sims)) / 2
        print(f"         → suggested threshold ≈ {midpoint:.3f}")
