import numpy as np
from core.matcher import Matcher

matcher = Matcher()

# Simulate a query (use one of your stored embeddings)
# For testing, just load one from database
sample = np.load("embeddings_db/prityanshu_body.npy")

person, score = matcher.identify(body_emb=sample)

print("Identified:", person)
print("Score:", score)
