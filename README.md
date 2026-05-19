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
```
(Can test with your own model of choice)

Visualization Dashboard

<img width="1840" height="635" alt="image" src="https://github.com/user-attachments/assets/c1af3d67-d0ba-47b7-bf3e-58af8ef1b7d0" />

<img width="1715" height="378" alt="image" src="https://github.com/user-attachments/assets/c8d8fb1c-b4df-4eb2-a7e9-dc59dc955c1c" />

<img width="1795" height="454" alt="image" src="https://github.com/user-attachments/assets/2c2e1709-71de-423a-9db3-e4fc80784009" />

<img width="1820" height="743" alt="image" src="https://github.com/user-attachments/assets/dfe1fb35-127e-4d91-a2d4-f38164def124" />

<img width="1770" height="725" alt="image" src="https://github.com/user-attachments/assets/374f2fa2-feae-4070-8e64-c455cde484c4" />

