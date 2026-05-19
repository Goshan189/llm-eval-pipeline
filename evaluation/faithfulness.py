from .model_loader import get_model
from sklearn.metrics.pairwise import cosine_similarity

def compute_faithfulness(context, response):
    if not context:
        return None

    model = get_model()
    emb1 = model.encode([context])
    emb2 = model.encode([response])

    score = cosine_similarity(emb1, emb2)[0][0]
    return float(round(score, 3))