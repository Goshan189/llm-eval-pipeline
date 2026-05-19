# 🤖 LLM Evaluation Pipeline with CI/CD-inspired Model Selection

A structured system to **evaluate, compare, and automatically select the best-performing LLM** based on accuracy, reliability, and latency.

---

## 🧠 Overview

This project implements an **end-to-end evaluation pipeline** for Large Language Models (LLMs), combining:

- **CI-style evaluation** → automated testing with metrics  
- **CD-style selection** → best model promotion based on performance  
- **Dashboard visualization** → clear comparison and insights  

---

## ⚙️ Features

### 🔹 Multi-Model Evaluation
Run and compare multiple local LLMs (via Ollama):

```bash
python run_pipeline.py --models llama3.1:8b gemma:7b tinyllama
