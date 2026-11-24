import time
import pandas as pd
from tracer import AgentRunTracker, AgentLoopException, BudgetExceededException


def simulate_agent_workflow(
    scenario: str = "Healthy Multi-Step Agent",
    model_name: str = "claude-sonnet-4",
    max_repeats: int = 3,
    max_budget_usd: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    tracker = AgentRunTracker(
        run_id=f"run_{int(time.time())}",
        model_name=model_name,
        max_repeats=max_repeats,
        max_budget_usd=max_budget_usd,
    )

    @tracker.trace_step("plan_task", tool_or_model=model_name)
    def plan_task(query: str):
        time.sleep(0.04)
        return "Plan: search_docs -> query_sql -> synthesize", 420, 145, f"Planned workflow for '{query}'"

    @tracker.trace_step("search_vector_db", tool_or_model="Tool:VectorSearch")
    def search_vector_db(topic: str):
        time.sleep(0.06)
        return "Retrieved 5 chunks", 1150, 210, f"Top-5 semantic retrieval for '{topic}'"

    @tracker.trace_step("query_sql_database", tool_or_model="Tool:SQLite")
    def query_sql_database(sql: str):
        time.sleep(0.03)
        return "Error: column 'rev_2026' not found", 680, 95, f"Executed query: {sql}"

    @tracker.trace_step("synthesize_answer", tool_or_model=model_name)
    def synthesize_answer(context: str):
        time.sleep(0.08)
        return "Final executive report generated.", 2400, 620, "Generated final user-facing response"

    outcome = "COMPLETED"
    alert_msg = "Agent completed all steps cleanly within budget and step limits."

    try:
        if scenario == "Healthy Multi-Step Agent":
            plan_task("Q3 AI Infrastructure Report")
            search_vector_db("GPU cluster utilization")
            query_sql_database("SELECT * FROM metrics_q3")
            synthesize_answer("Combined vector + SQL findings")

        elif scenario == "Agent Stuck in Tool Loop (Circuit Breaker)":
            plan_task("Fetch 2026 revenue breakdown")
            search_vector_db("revenue schema")
            # Agent stubbornly retries the exact same failing SQL query
            for _ in range(6):
                query_sql_database("SELECT rev_2026 FROM financials")

        elif scenario == "High-Context RAG Run (Budget Alert)":
            plan_task("Deep literature review across 20 papers")
            for i in range(6):
                search_vector_db(f"Research paper batch #{i + 1}")
            synthesize_answer("Synthesized 20 academic papers")

    except AgentLoopException as exc:
        outcome = "HALTED (LOOP DETECTED)"
        alert_msg = str(exc)
    except BudgetExceededException as exc:
        outcome = "HALTED (BUDGET CAP)"
        alert_msg = str(exc)

    rows = [
        {
            "Step": s.step_num,
            "Operation": s.step_name,
            "Engine / Tool": s.tool_or_model,
            "Latency (ms)": s.latency_ms,
            "Input Tokens": s.prompt_tokens,
            "Output Tokens": s.completion_tokens,
            "Cost ($)": s.cost_usd,
            "Status": s.status,
            "Details": s.payload_summary,
        }
        for s in tracker.steps
    ]

    summary = {
        "outcome": outcome,
        "alert": alert_msg,
        "total_steps": len(tracker.steps),
        "total_tokens": tracker.total_tokens,
        "total_cost": tracker.total_cost,
        "total_latency_ms": tracker.total_latency_ms,
    }
    return pd.DataFrame(rows), summary
