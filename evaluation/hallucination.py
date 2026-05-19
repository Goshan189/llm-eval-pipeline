def is_hallucinated(relevance_score, threshold=0.5):
    return relevance_score < threshold