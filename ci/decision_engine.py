def evaluate_run(metrics, mode="balanced"):

    relevance = metrics.get("avg_relevance", 0)
    hallucination = metrics.get("hallucination_rate", 100)
    faithfulness = metrics.get("avg_faithfulness", 0)
    latency = metrics.get("avg_latency", 5)

    latency_score = max(0, min(1, (5 - latency) / 5))

    # -----------------------
    # WEIGHTS
    # -----------------------
    if mode == "fast":
        weights = (0.3, 0.2, 0.1, 0.4)
    elif mode == "accurate":
        weights = (0.5, 0.3, 0.15, 0.05)
    else:
        weights = (0.4, 0.3, 0.2, 0.1)

    # -----------------------
    # SCORE
    # -----------------------
    score = (
        relevance * weights[0] * 100 +
        (1 - hallucination/100) * weights[1] * 100 +
        faithfulness * weights[2] * 100 +
        latency_score * weights[3] * 100
    )

    score = round(score, 2)

    status = "PASS" if score > 60 else "FAIL"

    # -----------------------
    # GRADE
    # -----------------------
    if score > 80:
        grade = "A"
    elif score > 65:
        grade = "B"
    elif score > 50:
        grade = "C"
    else:
        grade = "D"

    # -----------------------
    # INSIGHTS
    # -----------------------
    strengths = []
    weaknesses = []

    if relevance > 0.6:
        strengths.append("Good answer relevance")
    else:
        weaknesses.append("Low answer relevance")

    if hallucination < 30:
        strengths.append("Low hallucination")
    else:
        weaknesses.append("High hallucination")

    if faithfulness > 0.6:
        strengths.append("Good context understanding")
    else:
        weaknesses.append("Weak context grounding")

    if latency < 2:
        strengths.append("Fast responses")
    else:
        weaknesses.append("Slow response time")

    # -----------------------
    # SUMMARY
    # -----------------------
    summary = f"Model shows {('strong' if score > 60 else 'weak')} overall performance with score {score}."

    return {
        "status": status,
        "score": score,
        "mode": mode,
        "latency": latency,
        "grade": grade,
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses
    }