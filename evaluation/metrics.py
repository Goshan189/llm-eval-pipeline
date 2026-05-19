from collections import Counter

def compute_aggregate_metrics(results):
    total = len(results)

    relevance_scores = [r["relevance_score"] for r in results]
    hallucinations = [r["hallucinated"] for r in results]

    faithfulness_scores = [
        r["faithfulness_score"]
        for r in results
        if r["faithfulness_score"] is not None
    ]

    avg_relevance = sum(relevance_scores) / total
    hallucination_rate = (sum(hallucinations) / total) * 100

    avg_faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores else None
    )

    error_counts = Counter([r["error_type"] for r in results])

    return {
        "avg_relevance": round(avg_relevance, 3),
        "hallucination_rate": round(hallucination_rate, 2),
        "avg_faithfulness": round(avg_faithfulness, 3) if avg_faithfulness else None,
        "error_distribution": dict(error_counts)
    }