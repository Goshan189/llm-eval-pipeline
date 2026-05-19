import json
import os
from datetime import datetime

BASE_PATH = "cd"
REGISTRY_PATH = os.path.join(BASE_PATH, "registry", "model_registry.json")
PROD_PATH = os.path.join(BASE_PATH, "production")


def ensure_dirs():
    os.makedirs(os.path.join(BASE_PATH, "registry"), exist_ok=True)
    os.makedirs(PROD_PATH, exist_ok=True)


def load_registry():
    ensure_dirs()
    if not os.path.exists(REGISTRY_PATH):
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def save_registry(data):
    ensure_dirs()
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def log_model_run(model, score, decision):
    registry = load_registry()

    entry = {
        "model": model,
        "score": score,
        "status": decision.get("status"),
        "mode": decision.get("mode"),
        "latency": decision.get("latency"),
        "timestamp": str(datetime.now())
    }

    registry.append(entry)
    save_registry(registry)


def get_best_model(mode="balanced"):
    registry = load_registry()

    valid = [
        r for r in registry
        if r["status"] == "PASS" and r["mode"] == mode
    ]

    if not valid:
        return None

    return max(valid, key=lambda x: (x["score"], -x["latency"]))


def promote_best_model(mode="balanced"):
    ensure_dirs()

    best = get_best_model(mode)

    if not best:
        return None

    path = os.path.join(PROD_PATH, f"production_{mode}.json")

    with open(path, "w") as f:
        json.dump(best, f, indent=2)

    return best