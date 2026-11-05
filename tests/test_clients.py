import json

from google.genai import types
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict

from bench import clients
from bench.schema import StorageLabel


class FakeOpenAI:
    """Підроблений OpenAI-клієнт: повертає відповіді й запам'ятовує запити."""

    def __init__(self, *messages):
        self.replies, self.requests = list(messages), []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = self.replies.pop(0)
        return ChatCompletion.model_validate({
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70,
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        })


class FakeGemini:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        return self.responses.pop(0)


def use_fake(monkeypatch, fake):
    monkeypatch.setattr(clients, "_client", lambda spec: fake)
    return fake


def test_native_client_has_timeout_and_no_sdk_retries(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(clients.genai, "Client", fake_client)
    clients._clients.pop("gemini-native", None)
    try:
        clients._client(clients.models.get("gemini-native"))
    finally:
        clients._clients.pop("gemini-native", None)
    options = captured["http_options"]
    assert options.timeout == 180_000
    assert options.retry_options.attempts == 1


def gemini_response(*parts):
    return types.GenerateContentResponse(
        model_version="fake-gemini",
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=list(parts)),
            finish_reason="STOP",
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=40,
            candidates_token_count=10,
            thoughts_token_count=30,
        ),
    )


def test_plain_text_usage_and_parameter_override(monkeypatch):
    fake = use_fake(monkeypatch, FakeOpenAI({"role": "assistant", "content": "redis"}))
    turn = clients.chat("local", [{"role": "user", "content": "x"}], temperature=1)
    assert turn.text == "redis" and turn.schema_valid is None and turn.tool_calls == []
    assert (turn.usage.input, turn.usage.output, turn.usage.reasoning) == (50, 20, 12)
    assert fake.requests[0]["temperature"] == 1
    assert fake.requests[0]["seed"] == 7
    assert fake.requests[0]["extra_body"]["num_ctx"] == 16384


def test_schema_valid_and_strict_request(monkeypatch):
    fake = use_fake(
        monkeypatch,
        FakeOpenAI({"role": "assistant", "content": '{"label": "postgresql"}'}),
    )
    turn = clients.chat("local", [{"role": "user", "content": "x"}], schema=StorageLabel)
    response_format = fake.requests[0]["response_format"]
    assert turn.schema_valid is True and turn.parsed.label == "postgresql"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False


def test_schema_invalid_extra_field(monkeypatch):
    use_fake(
        monkeypatch,
        FakeOpenAI({"role": "assistant", "content": '{"label": "redis", "why": "бо"}'}),
    )
    turn = clients.chat("local", [{"role": "user", "content": "x"}], schema=StorageLabel)
    assert turn.schema_valid is False and turn.parsed is None


def test_invalid_tool_arguments_are_recorded(monkeypatch):
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{oops"},
    }
    use_fake(monkeypatch, FakeOpenAI({"role": "assistant", "content": None, "tool_calls": [call]}))
    turn = clients.chat("local", [{"role": "user", "content": "x"}])
    assert turn.tool_calls[0].arguments is None
    assert turn.tool_calls[0].parse_error


def test_unknown_extra_content_is_passed_back_unchanged(monkeypatch):
    extra = {"google": {"thought_signature": "c2lnbmF0dXJl"}}
    call = {
        "id": "c1",
        "type": "function",
        "extra_content": extra,
        "function": {"name": "lookup", "arguments": '{"id": "A-1"}'},
    }
    fake = use_fake(
        monkeypatch,
        FakeOpenAI(
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "assistant", "content": '{"label": "redis"}'},
        ),
    )
    history = [{"role": "user", "content": "x"}]
    turn = clients.chat("local", history)
    history += [
        turn.raw_assistant_message,
        {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "ok"},
    ]
    clients.chat("local", history)
    assert fake.requests[1]["messages"][1]["tool_calls"][0]["extra_content"] == extra


def test_to_strict_schema_fixes_nested_objects():
    class Detail(BaseModel):
        model_config = ConfigDict(extra="forbid")
        note: str | None

    class Envelope(BaseModel):
        model_config = ConfigDict(extra="forbid")
        label: str
        detail: Detail

    strict = clients.to_strict_schema(Envelope)
    detail = strict["$defs"]["Detail"]
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["label", "detail"]
    assert detail["additionalProperties"] is False
    assert detail["required"] == ["note"]


def test_gemini_manual_tool_loop_keeps_content(monkeypatch):
    call_part = types.Part(
        function_call=types.FunctionCall(id="f1", name="lookup", args={"id": "A-1"}),
        thought_signature=b"sig",
    )
    fake = use_fake(
        monkeypatch,
        FakeGemini(
            gemini_response(call_part),
            gemini_response(types.Part(text=json.dumps({"label": "redis"}))),
        ),
    )
    tools = [{"type": "function", "function": {
        "name": "lookup",
        "description": "d",
        "parameters": {"type": "object", "properties": {}},
    }}]
    history = [{"role": "system", "content": "Класифікуй."}, {"role": "user", "content": "x"}]
    turn = clients.chat("gemini-native", history, tools=tools)
    assert turn.tool_calls[0].arguments == {"id": "A-1"}
    assert (turn.usage.output, turn.usage.reasoning) == (40, 30)
    assert fake.requests[0]["config"].automatic_function_calling.disable is True
    history += [
        turn.raw_assistant_message,
        {"role": "tool", "tool_call_id": "f1", "name": "lookup", "content": "ok"},
    ]
    final = clients.chat("gemini-native", history)
    contents = fake.requests[1]["contents"]
    assert contents[1] is turn.raw_assistant_message["gemini_content"]
    assert contents[1].parts[0].thought_signature == b"sig"
    assert contents[2].parts[0].function_response.id == "f1"
    assert final.text == '{"label": "redis"}' and final.finish_reason == "stop"
