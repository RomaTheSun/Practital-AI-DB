import json

from bench.agent_loop import run_agent
from bench.clients import ToolCall, Turn, Usage

USER = [{"role": "user", "content": "Що вибрати для W-101?"}]


def turn(text=None, calls=(), usage=(10, 5), finish_reason=None):
    tool_calls = [ToolCall(i, n, raw, None, None) for i, n, raw in calls]
    raw = {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": n, "arguments": a}}
            for i, n, a in calls
        ],
    }
    return Turn(
        text,
        None,
        None,
        tool_calls,
        Usage(*usage),
        finish_reason or ("tool_calls" if calls else "stop"),
        1,
        "fake",
        raw,
    )


class FakeChat:
    def __init__(self, *turns):
        self.turns, self.histories = list(turns), []

    def __call__(self, alias, messages, tools=None, **params):
        self.histories.append(list(messages))
        return self.turns.pop(0) if len(self.turns) > 1 else self.turns[0]


def test_calls_tool_then_answers(tmp_path):
    chat = FakeChat(
        turn(calls=[("c1", "lookup_workload", '{"code": "W-101"}')]),
        turn('{"label": "postgresql"}'),
    )
    trace = tmp_path / "trace.jsonl"
    result = run_agent(
        "local",
        USER,
        ["lookup_workload"],
        chat=chat,
        run_id="r",
        case_id="k01",
        trace_path=str(trace),
    )
    assert (result.stop_reason, result.steps, result.failures, result.tools_called) == (
        "final",
        2,
        [],
        ["lookup_workload"],
    )
    assert result.final_text == '{"label": "postgresql"}'
    assert result.usage_total.input == 20
    tool_message = chat.histories[1][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c1"
    assert "атомарно змінювати" in tool_message["content"]
    lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [line["tool"] for line in lines] == ["lookup_workload", None]
    assert lines[0]["case_id"] == "k01"


def test_stops_on_max_steps():
    chat = FakeChat(
        *[
            turn(calls=[(f"c{i}", "lookup_workload", json.dumps({"code": f"W-{100 + i}"}))])
            for i in range(5)
        ]
    )
    result = run_agent("local", USER, ["lookup_workload"], chat=chat, max_steps=3)
    assert result.stop_reason == "max_steps" and result.steps == 3
    assert result.failures == ["max_steps"] and result.final_text is None


def test_stops_on_token_budget():
    chat = FakeChat(turn(calls=[("c1", "lookup_workload", '{"code":"W-101"}')], usage=(15000, 6000)))
    result = run_agent(
        "local", USER, ["lookup_workload"], chat=chat, token_budget=20000
    )
    assert result.stop_reason == "budget" and result.steps == 1
    assert "budget" in result.failures and len(chat.histories) == 1


def test_unknown_tool_does_not_break_loop():
    chat = FakeChat(
        turn(calls=[("c1", "does_not_exist", "{}")]),
        turn('{"label":"postgresql"}'),
    )
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "final" and result.failures == ["unknown_tool"]
    assert chat.histories[1][-1]["content"].startswith("unknown tool")


def test_bad_arguments_do_not_break_loop():
    chat = FakeChat(
        turn(calls=[("c1", "lookup_workload", '{"id": 5}')]),
        turn('{"label":"postgresql"}'),
    )
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "final" and result.failures == ["bad_args"]
    assert chat.histories[1][-1]["content"].startswith("bad arguments")


def test_two_parallel_calls_keep_ids_and_order():
    chat = FakeChat(
        turn(
            calls=[
                ("c1", "lookup_workload", '{"code":"W-105"}'),
                ("c2", "lookup_requirement", '{"code":"R-205"}'),
            ]
        ),
        turn('{"label":"postgresql"}'),
    )
    result = run_agent(
        "local", USER, ["lookup_workload", "lookup_requirement"], chat=chat
    )
    messages = chat.histories[1][-2:]
    assert [message["tool_call_id"] for message in messages] == ["c1", "c2"]
    assert result.tools_called == ["lookup_workload", "lookup_requirement"]


def test_repeated_call_stops_on_third_occurrence():
    repeated = turn(calls=[("c1", "lookup_workload", '{"code":"W-101"}')])
    chat = FakeChat(repeated)
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "repeated_call" and result.steps == 3
    assert result.failures == ["repeated_call"]


def test_tool_error_reaches_final_answer():
    chat = FakeChat(
        turn(calls=[("c1", "lookup_workload", '{"code":"W-XYZ"}')]),
        turn('{"label":"postgresql"}'),
    )
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "final" and result.failures == ["tool_error"]
    assert chat.histories[1][-1]["content"].startswith("tool error")


def test_tool_call_in_text_is_classified():
    chat = FakeChat(turn('{"name":"lookup_workload","arguments":{"code":"W-101"}}'))
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "final" and result.failures == ["tool_call_in_text"]


def test_provider_tool_failure_reason_is_not_final():
    chat = FakeChat(turn(finish_reason="malformed_function_call"))
    result = run_agent("local", USER, ["lookup_workload"], chat=chat)
    assert result.stop_reason == "malformed_function_call"
    assert result.failures == ["malformed_function_call"]
