from sklearn.metrics.pairwise import cosine_similarity
from .model_loader import get_model

def compute_relevance(expected, response):
    model = get_model()

    emb1 = model.encode([expected])
    emb2 = model.encode([response])

    score = cosine_similarity(emb1, emb2)[0][0]
    return float(round(score, 3))