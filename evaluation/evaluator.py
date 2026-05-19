import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from .relevance import compute_relevance
from .faithfulness import compute_faithfulness
from .hallucination import is_hallucinated
from .metrics import compute_aggregate_metrics
from .error_analysis import detect_error_type, is_verbose
from ci.decision_engine import evaluate_run


def evaluate_results(input_path, mode="balanced"):
    with open(input_path, "r") as f:
        data = json.load(f)

    evaluated = []

    for item in data["results"]:

        response = item["response"]
        expected = item["expected_answer"]
        context = item.get("context")

        relevance = compute_relevance(expected, response)
        faithfulness = compute_faithfulness(context, response)
        hallucinated = is_hallucinated(relevance)

        error_type = detect_error_type(response, expected, context)
        verbose_flag = is_verbose(response)

        evaluated.append({
            **item,
            "relevance_score": relevance,
            "faithfulness_score": faithfulness,
            "hallucinated": hallucinated,
            "error_type": error_type,
            "is_verbose": verbose_flag
        })

    # -----------------------
    # METRICS
    # -----------------------
    metrics = compute_aggregate_metrics(evaluated)

    # DECISION (mode aware)
    decision = evaluate_run(metrics, mode=mode)

    # -----------------------
    # SAVE OUTPUTS
    # -----------------------
    run_name = os.path.basename(input_path).replace(".json", "")

    os.makedirs("evaluation/responses", exist_ok=True)
    os.makedirs("evaluation/metrics", exist_ok=True)

    responses_path = f"evaluation/responses/{run_name}_responses.json"
    metrics_path = f"evaluation/metrics/{run_name}_metrics.json"

    with open(responses_path, "w") as f:
        json.dump({"model": data["model"], "evaluated_results": evaluated}, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump({
            "model": data["model"],
            "metrics": metrics,
            "decision": decision
        }, f, indent=2)

    return responses_path, metrics_path