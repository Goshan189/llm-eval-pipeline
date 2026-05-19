import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="LLM Comparison Dashboard", layout="wide")

# -----------------------
# LOAD METRICS + DECISION
# -----------------------
def load_all_data():
    files = [f for f in os.listdir("evaluation/metrics") if f.endswith(".json")]

    rows = []
    decisions = {}

    for f_name in files:
        with open(f"evaluation/metrics/{f_name}") as f:
            d = json.load(f)

            model = d.get("model", f_name)

            acc = d["metrics"]["avg_relevance"]
            hall = d["metrics"]["hallucination_rate"]
            ctx = d["metrics"]["avg_faithfulness"] or 0

            rows.append({
                "model": model,
                "accuracy": acc,
                "risk": hall,
                "context": ctx
            })

            decisions[model] = d.get("decision", {})

    df = pd.DataFrame(rows)

    # -----------------------
    # NORMALIZATION
    # -----------------------
    df["accuracy_score"] = df["accuracy"] * 100
    df["reliability_score"] = 100 - df["risk"]
    df["context_score"] = df["context"] * 100

    return df, decisions


# -----------------------
# LOAD RESPONSES
# -----------------------
def load_responses():
    files = [f for f in os.listdir("evaluation/responses") if f.endswith(".json")]
    data = {}

    for f_name in files:
        with open(f"evaluation/responses/{f_name}") as f:
            d = json.load(f)
            data[d["model"]] = d["evaluated_results"]

    return data


df, decision_data = load_all_data()
responses = load_responses()

if df.empty:
    st.error("No data found. Run evaluation first.")
    st.stop()

# -----------------------
# HEADER
# -----------------------
st.title("🤖 LLM Comparison Dashboard")
st.caption("Understand model performance, reliability, and behavior")

# -----------------------
# BEST MODEL (BASED ON SCORE)
# -----------------------
def get_score(model):
    return decision_data.get(model, {}).get("score", 0)

df["final_score"] = df["model"].apply(get_score)

best = df.sort_values("final_score", ascending=False).iloc[0]
best_decision = decision_data.get(best["model"], {})

# -----------------------
# BEST MODEL CARD
# -----------------------
st.markdown("## 🏆 Best Model")

st.markdown(f"""
<div style='padding:20px;
border-radius:12px;
background: linear-gradient(135deg, #1f2937, #111827);
color: white;
border:1px solid #333;
box-shadow: 0 4px 20px rgba(0,0,0,0.3)'>

<h2>{best['model']}</h2>

<p><b>Score:</b> {best_decision.get("score", "-")} / 100</p>
<p><b>Grade:</b> {best_decision.get("grade", "-")}</p>
<p><b>Status:</b> {best_decision.get("status", "-")}</p>

<p><b>Summary:</b> {best_decision.get("summary", "-")}</p>

</div>
""", unsafe_allow_html=True)

# -----------------------
# STRENGTHS / WEAKNESSES
# -----------------------
st.markdown("### 🔍 Why this model?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Strengths**")
    for s in best_decision.get("strengths", []):
        st.success(s)

with col2:
    st.markdown("**Weaknesses**")
    for w in best_decision.get("weaknesses", []):
        st.error(w)

st.markdown("---")

# -----------------------
# MODEL COMPARISON TABLE
# -----------------------
st.markdown("## ⚖️ Compare Models")

def level(x):
    if x > 70: return "High"
    elif x > 40: return "Medium"
    else: return "Low"

display = df.copy()
display["Accuracy"] = display["accuracy_score"].apply(level)
display["Reliability"] = display["reliability_score"].apply(level)
display["Context"] = display["context_score"].apply(level)

display["Score"] = display["final_score"]

st.dataframe(display[[
    "model", "Accuracy", "Reliability", "Context", "Score"
]])

st.markdown("---")

# -----------------------
# PERFORMANCE MAP
# -----------------------
st.markdown("## 📊 Model Performance Map")
st.caption("Top-right = best (accurate + reliable)")

fig = px.scatter(
    df,
    x="accuracy_score",
    y="reliability_score",
    size="context_score",
    color="model",
    hover_name="model",
    labels={
        "accuracy_score": "Accuracy (0–100)",
        "reliability_score": "Reliability (0–100)"
    }
)

fig.update_layout(
    xaxis=dict(range=[0, 100]),
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------
# MODEL INSIGHTS (ALL MODELS)
# -----------------------
st.markdown("## 🧠 Model Insights")

for _, row in df.iterrows():
    model = row["model"]
    d = decision_data.get(model, {})

    with st.expander(f"{model} — Score: {d.get('score', '-')}, Grade: {d.get('grade', '-')}"):
        st.write("**Status:**", d.get("status", "-"))
        st.write("**Summary:**", d.get("summary", "-"))

        st.write("**Strengths:**")
        for s in d.get("strengths", []):
            st.success(s)

        st.write("**Weaknesses:**")
        for w in d.get("weaknesses", []):
            st.error(w)

st.markdown("---")

# -----------------------
# RESPONSE EXPLORER
# -----------------------
st.markdown("## 🔍 Explore Model Behavior")

model_choice = st.selectbox("Select Model", df["model"])

model_responses = responses.get(model_choice, [])

if model_responses:
    df_resp = pd.DataFrame(model_responses)

    error_filter = st.selectbox(
        "Filter by Behavior",
        ["All"] + list(df_resp["error_type"].unique())
    )

    if error_filter != "All":
        df_resp = df_resp[df_resp["error_type"] == error_filter]

    st.dataframe(df_resp[[
        "question",
        "response",
        "expected_answer",
        "error_type"
    ]])
else:
    st.warning("No responses available.")

st.markdown("---")

# -----------------------
# GLOBAL INSIGHTS
# -----------------------
st.markdown("## 🧠 Overall Insights")

avg_risk = df["risk"].mean()

if avg_risk > 50:
    st.error("Most models show high hallucination (low reliability).")
elif avg_risk > 20:
    st.warning("Moderate reliability across models.")
else:
    st.success("Models are generally reliable.")

st.write(f"Total Models Compared: {len(df)}")