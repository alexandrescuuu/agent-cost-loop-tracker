import time
import functools
from dataclasses import dataclass, field
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# Configure OpenTelemetry Tracer
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
otel_tracer = trace.get_tracer("agent.observability")

# Pricing per 1M tokens (USD)
MODEL_PRICING = {
    "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama-3.3-70b": {"input": 0.59, "output": 0.79},
}


class AgentLoopException(RuntimeError):
    """Raised by the Circuit Breaker when an agent repeats the same step infinitely."""


class BudgetExceededException(RuntimeError):
    """Raised when cumulative agent run cost exceeds the configured budget cap."""


@dataclass
class StepMetric:
    step_num: int
    step_name: str
    tool_or_model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    status: str
    payload_summary: str


@dataclass
class AgentRunTracker:
    run_id: str
    model_name: str = "claude-sonnet-4"
    max_repeats: int = 3
    max_budget_usd: float = 0.05
    steps: list[StepMetric] = field(default_factory=list)
    _call_signatures: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        rates = MODEL_PRICING.get(self.model_name, MODEL_PRICING["claude-sonnet-4"])
        in_cost = (prompt_tokens / 1_000_000) * rates["input"]
        out_cost = (completion_tokens / 1_000_000) * rates["output"]
        return in_cost + out_cost

    def trace_step(self, step_name: str, tool_or_model: str = "LLM"):
        """Decorator wrapping any agent reasoning step or tool call with OpenTelemetry + Loop & Budget guards."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                sig = f"{step_name}:{args}:{sorted(kwargs.items())}"
                self._call_signatures.append(sig)
                step_idx = len(self.steps) + 1

                # 1. Loop Detection Circuit-Breaker
                recent = self._call_signatures[-self.max_repeats:]
                if len(recent) == self.max_repeats and len(set(recent)) == 1:
                    self.steps.append(
                        StepMetric(
                            step_num=step_idx,
                            step_name=step_name,
                            tool_or_model=tool_or_model,
                            latency_ms=0.0,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost_usd=0.0,
                            status="LOOP_HALTED",
                            payload_summary=f"Circuit breaker tripped after {self.max_repeats} identical calls",
                        )
                    )
                    raise AgentLoopException(
                        f"Infinite loop detected on '{step_name}' (repeated {self.max_repeats}x). Execution halted."
                    )

                # 2. Execute inside an OpenTelemetry Span
                with otel_tracer.start_as_current_span(step_name) as span:
                    span.set_attribute("agent.run_id", self.run_id)
                    span.set_attribute("agent.step", step_idx)
                    span.set_attribute("gen_ai.system", tool_or_model)

                    t0 = time.perf_counter()
                    result, p_tok, c_tok, summary = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    step_cost = self.estimate_cost(p_tok, c_tok)

                    span.set_attribute("gen_ai.usage.input_tokens", p_tok)
                    span.set_attribute("gen_ai.usage.output_tokens", c_tok)
                    span.set_attribute("agent.cost_usd", step_cost)

                    self.steps.append(
                        StepMetric(
                            step_num=step_idx,
                            step_name=step_name,
                            tool_or_model=tool_or_model,
                            latency_ms=round(elapsed_ms, 2),
                            prompt_tokens=p_tok,
                            completion_tokens=c_tok,
                            cost_usd=round(step_cost, 6),
                            status="OK",
                            payload_summary=summary,
                        )
                    )

                    # 3. Budget Guardrail
                    if self.total_cost > self.max_budget_usd:
                        raise BudgetExceededException(
                            f"Run budget ${self.max_budget_usd:.4f} exceeded (current: ${self.total_cost:.4f})."
                        )
                    return result

            return wrapper
        return decorator
