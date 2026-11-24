import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tracer import MODEL_PRICING
from agent_sim import simulate_agent_workflow

st.set_page_config(
    page_title="Agent Cost, Latency & Loop Observability Tracker",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Agent Observability: Cost, Latency & Loop Circuit-Breaker")
st.markdown(
    "Lightweight **OpenTelemetry** instrumentation wrapper for autonomous AI agents — tracking **per-step token usage**, "
    "**multi-model USD cost attribution**, **span latency**, and **automatic infinite-loop termination**."
)

# Sidebar controls
st.sidebar.header("Agent & Guardrail Settings")
scenario = st.sidebar.selectbox(
    "Select Agent Execution Scenario",
    (
        "Agent Stuck in Tool Loop (Circuit Breaker)",
        "Healthy Multi-Step Agent",
        "High-Context RAG Run (Budget Alert)",
    ),
)
model_name = st.sidebar.selectbox("LLM Pricing Profile", tuple(MODEL_PRICING.keys()))
max_repeats = st.sidebar.slider("Loop Circuit-Breaker Threshold (Max Repeats)", min_value=2, max_value=5, value=3)
max_budget = st.sidebar.slider("Max Run Budget Cap ($ USD)", min_value=0.005, max_value=0.100, value=0.045, step=0.005, format="%.3f")

df_steps, summary = simulate_agent_workflow(
    scenario=scenario,
    model_name=model_name,
    max_repeats=max_repeats,
    max_budget_usd=max_budget,
)

# Export OpenTelemetry Trace JSON in Sidebar
st.sidebar.divider()
st.sidebar.subheader("OpenTelemetry Export")
otel_payload = {
    "resource": {"service.name": "ai-agent-observability", "gen_ai.system": model_name},
    "summary": summary,
    "spans": df_steps.to_dict(orient="records"),
}
st.sidebar.download_button(
    label="Download OTel Trace (.json)",
    data=json.dumps(otel_payload, indent=2),
    file_name="agent_otel_trace.json",
    mime="application/json",
)

# Status Banner
if summary["outcome"] == "COMPLETED":
    st.success(f"**{summary['outcome']}:** {summary['alert']}")
else:
    st.error(f"**{summary['outcome']}:** {summary['alert']}")

# Calculate ROI / Cost Saved if halted early
unprotected_est_cost = summary["total_cost"] * 2.8 if "HALTED" in summary["outcome"] else summary["total_cost"]
saved_usd = max(0.0, unprotected_est_cost - summary["total_cost"])

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Executed Steps", summary["total_steps"])
col2.metric("Total Tokens", f"{summary['total_tokens']:,}")
col3.metric("Total Run Cost", f"${summary['total_cost']:.5f}")
col4.metric("Guardrail Savings", f"${saved_usd:.5f}", delta="Loop Prevented" if saved_usd > 0 else "Optimal")
col5.metric("Total Latency", f"{summary['total_latency_ms']:.1f} ms")

# Charts Row
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("Cost Attribution per Agent Step ($ USD)")
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.6), dpi=120)
    bar_colors = ["#FF4B4B" if st_val != "OK" else "#1f77b4" for st_val in df_steps["Status"]]
    labels = [f"S{r['Step']}: {r['Operation']}" for _, r in df_steps.iterrows()]
    ax1.bar(labels, df_steps["Cost ($)"], color=bar_colors)
    ax1.set_ylabel("Step Cost ($ USD)")
    ax1.set_title(f"Token Cost Breakdown ({model_name})")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig1)

with c_right:
    st.subheader("Step Latency Waterfall (ms)")
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.6), dpi=120)
    ax2.barh(labels[::-1], df_steps["Latency (ms)"][::-1], color="#ff7f0e")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_title("OpenTelemetry Span Duration per Step")
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig2)

# Multi-Model Cost Comparison Section
st.subheader("Multi-Model Cost Projection for This Exact Agent Run")
total_in = int(df_steps["Input Tokens"].sum())
total_out = int(df_steps["Output Tokens"].sum())

model_comp = []
for m_name, rates in MODEL_PRICING.items():
    projected = (total_in / 1_000_000) * rates["input"] + (total_out / 1_000_000) * rates["output"]
    model_comp.append({
        "Model": m_name,
        "Input Rate ($/1M)": rates["input"],
        "Output Rate ($/1M)": rates["output"],
        "Projected Run Cost ($)": round(projected, 6),
        "Cost for 10,000 Runs ($)": round(projected * 10_000, 2),
    })

df_models = pd.DataFrame(model_comp).sort_values("Projected Run Cost ($)", ascending=False)
m_chart, m_table = st.columns((1.1, 1.3))

with m_chart:
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    colors_m = ["#FF4B4B" if m == model_name else "#2ca02c" for m in df_models["Model"]]
    ax3.barh(df_models["Model"], df_models["Cost for 10,000 Runs ($)"], color=colors_m)
    ax3.set_xlabel("Projected Cost at 10k Scale ($ USD)")
    ax3.set_title("Production Scale Cost Comparison (10,000 Agent Runs)")
    ax3.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig3)

with m_table:
    st.dataframe(df_models, use_container_width=True)

st.subheader("OpenTelemetry Execution Trace Log")
st.dataframe(df_steps, use_container_width=True)
