from core.fusion_engine import FusionEngine

fusion = FusionEngine()

# Case 1: all modalities available
score1 = fusion.compute_final_score(face_score=0.82, body_score=0.76, gait_score=0.70, attr_score=0.60)

print("Case 1 Score:", score1)
print("Match:", fusion.decision(score1))


# Case 2: no face available
score2 = fusion.compute_final_score(face_score=None, body_score=0.80, gait_score=0.75, attr_score=None)

print("\nCase 2 Score:", score2)
print("Match:", fusion.decision(score2))
