import json
import time
import os
import requests
from datetime import datetime
from tqdm import tqdm

# -----------------------
# CONFIG
# -----------------------
OLLAMA_URL = "http://localhost:11434/api/generate"


# -----------------------
# Prompt Builder
# -----------------------
def build_prompt(question, context):
    base = (
        "You are an evaluation assistant.\n\n"

        "RULES:\n"
        "1. Answer in MAXIMUM 2 sentences.\n"
        "2. Do NOT hallucinate.\n"
        "3. If context is provided, use ONLY that context.\n"
        "4. If context is required but missing, reply exactly:\n"
        "'The context does not contain this information.'\n\n"

        "Answer must be clear and concise.\n\n"
    )

    if context:
        return (
            base +
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )
    else:
        return (
            base +
            f"Question:\n{question}\n\n"
            "Answer:"
        )


# -----------------------
# Output Cleaner
# -----------------------
def clean_output(answer):
    if not answer:
        return "ERROR: Empty response"

    answer = answer.strip()

    if "Note:" in answer:
        answer = answer.split("Note:")[0].strip()

    sentences = answer.split(".")
    answer = ".".join(sentences[:2]).strip()

    return answer


# -----------------------
# Ollama Query
# -----------------------
def query_ollama(prompt, model_name):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            },
            timeout=120
        )

        data = response.json()

        if "error" in data:
            return f"ERROR: {data['error']}"

        if "response" not in data:
            return f"ERROR: Unexpected response {data}"

        return data["response"].strip()

    except Exception as e:
        return f"ERROR: {str(e)}"


# -----------------------
# Runner (FIXED)
# -----------------------
def run_llm(dataset_path, model_name):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # 🔥 Safe run naming per model
    safe_model = model_name.replace(":", "_")
    run_id = f"{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = []

    print(f"\nRunning model: {model_name}")
    print("Total questions:", len(dataset), "\n")

    for item in tqdm(dataset, desc=f"{model_name}"):

        prompt = build_prompt(item["question"], item.get("context"))

        start = time.time()

        # Retry logic
        answer = ""
        for _ in range(2):
            answer = query_ollama(prompt, model_name)
            if answer and "ERROR" not in answer:
                break

        answer = clean_output(answer)

        latency = round(time.time() - start, 3)

        results.append({
            "id": item["id"],
            "question": item["question"],
            "context": item.get("context"),
            "expected_answer": item["expected_answer"],
            "response": answer,
            "latency": latency,
            "model": model_name,
            "tags": item["tags"]
        })

        time.sleep(0.3)

    # -----------------------
    # Save Results
    # -----------------------
    os.makedirs("data/results", exist_ok=True)

    output_path = f"data/results/{run_id}.json"

    with open(output_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "model": model_name,
            "timestamp": str(datetime.now()),
            "total_cases": len(results),
            "results": results
        }, f, indent=2)

    print("\nRun complete")
    print("Saved to:", output_path)

    return output_path


# -----------------------
# Optional standalone run
# -----------------------
if __name__ == "__main__":
    run_llm("data/dataset.json", "llama3.1:8b")