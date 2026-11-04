"""Ручний цикл виклику інструментів з явними умовами зупинки і трасою."""
from dataclasses import dataclass
from typing import Callable

from bench import clients, data
from bench.clients import ToolCall, Turn, Usage
from bench.tools import execute, tool_specs

TOOL_FAILURE_REASONS = (
    "malformed_function_call",
    "unexpected_tool_call",
    "too_many_tool_calls",
    "missing_thought_signature",
    "malformed_response",
)


@dataclass
class AgentResult:
    final_text: str | None
    stop_reason: str
    steps: int
    usage_total: Usage
    failures: list[str]
    tools_called: list[str]


def looks_like_tool_call(text: str | None) -> bool:
    """Модель надрукувала виклик замість структурованого tool_calls."""
    return bool(text) and (
        "[TOOL_REQUEST]" in text or ('"name"' in text and '"arguments"' in text)
    )


def add_usage(total: Usage, usage: Usage) -> None:
    for name in ("input", "output", "reasoning", "cached"):
        setattr(total, name, (getattr(total, name) or 0) + (getattr(usage, name) or 0))


def run_agent(
    alias: str,
    messages: list[dict],
    tool_names: list[str],
    *,
    max_steps: int = 6,
    token_budget: int = 20000,
    run_id: str = "",
    case_id: str = "",
    trace_path: str | None = None,
    chat: Callable[..., Turn] = clients.chat,
) -> AgentResult:
    history = list(messages)
    specs = tool_specs(tool_names) or None
    total = Usage(0, 0, 0, 0)
    failures: list[str] = []
    called: list[str] = []
    recent: list[tuple[str, str]] = []

    def trace(
        step: int,
        turn: Turn,
        call: ToolCall | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        if trace_path:
            data.append_row(
                trace_path,
                {
                    "ts": data.utc_now(),
                    "run_id": run_id,
                    "case_id": case_id,
                    "step": step,
                    "alias": alias,
                    "tool": call.name if call else None,
                    "tool_call_id": call.id if call else None,
                    "args_raw": call.arguments_raw if call else None,
                    "result": result,
                    "error_type": error,
                    "input_tokens": turn.usage.input,
                    "output_tokens": turn.usage.output,
                    "finish_reason": turn.finish_reason,
                },
            )

    def stop(reason: str, step: int, text: str | None = None) -> AgentResult:
        if reason != "final":
            failures.append(reason)
        return AgentResult(text, reason, step, total, failures, called)

    for step in range(1, max_steps + 1):
        turn = chat(alias, history, tools=specs)
        history.append(turn.raw_assistant_message)
        add_usage(total, turn.usage)
        if not turn.tool_calls:
            if turn.finish_reason in TOOL_FAILURE_REASONS:
                trace(step, turn, error=turn.finish_reason)
                return stop(turn.finish_reason, step, turn.text)
            trace(step, turn, result=turn.text)
            if looks_like_tool_call(turn.text):
                failures.append("tool_call_in_text")
            return stop("final", step, turn.text)
        if (total.input or 0) + (total.output or 0) > token_budget:
            trace(step, turn, error="budget")
            return stop("budget", step)
        for call in turn.tool_calls:
            called.append(call.name)
            recent.append((call.name, call.arguments_raw))
            if len(recent) >= 3 and len(set(recent[-3:])) == 1:
                trace(step, turn, call, error="repeated_call")
                return stop("repeated_call", step)
            result, error = execute(call.name, call.arguments_raw)
            if error:
                failures.append(error)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": result,
                }
            )
            trace(step, turn, call, result, error)
    return stop("max_steps", max_steps)
