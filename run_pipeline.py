import argparse
import json

from runner.llm_runner import run_llm
from evaluation.evaluator import evaluate_results
from cd.model_registry import log_model_run, promote_best_model


# -----------------------
# CLI ARGUMENTS
# -----------------------
parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")

parser.add_argument(
    "--models",
    nargs="+",
    required=True,
    help="List of models to evaluate"
)

parser.add_argument(
    "--mode",
    choices=["balanced", "fast", "accurate"],
    default="balanced",
    help="Optimization mode"
)

args = parser.parse_args()

MODELS = args.models
MODE = args.mode
DATASET_PATH = "data/dataset.json"


print("\n🚀 Running LLM Evaluation Pipeline")
print(f"Mode: {MODE}\n")


# -----------------------
# RUN PIPELINE
# -----------------------
for model_name in MODELS:

    print("\n==============================")
    print(f"🔹 MODEL: {model_name}")
    print("==============================")

    # STEP 1: RUN MODEL
    print("[1] Generating responses...")
    run_file = run_llm(DATASET_PATH, model_name)

    # STEP 2: EVALUATE
    print("[2] Evaluating...")
    responses_path, metrics_path = evaluate_results(run_file, mode=MODE)

    # STEP 3: READ DECISION
    with open(metrics_path, "r") as f:
        data = json.load(f)

    decision = data.get("decision", {})

    print("\n🧠 RESULT")
    print(f"Status : {decision.get('status')}")
    print(f"Score  : {decision.get('score')}")
    print(f"Mode   : {decision.get('mode')}")
    print(f"Latency: {decision.get('latency')}")

    # STEP 4: LOG TO REGISTRY
    log_model_run(
        model=model_name,
        score=decision.get("score", 0),
        decision=decision
    )


# -----------------------
# FINAL STEP: PROMOTION
# -----------------------
print("\n==============================")
print("🏆 Selecting Best Model...")
print("==============================")

best = promote_best_model(mode=MODE)

if best:
    print("\n✅ PRODUCTION MODEL UPDATED")
    print(f"Model  : {best.get('model')}")
    print(f"Score  : {best.get('score')}")
    print(f"Mode   : {best.get('mode')}")
    print(f"Latency: {best.get('latency')}")
else:
    print("\n⚠️ No valid model found for promotion")


print("\n🎉 Pipeline complete!\n")