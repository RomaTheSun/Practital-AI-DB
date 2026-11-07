"""Клієнт LLM v2 (крок 04): одна функція chat() над двома формами API.

Історія — повідомлення в стилі OpenAI. Відповідь асистента кладіть в історію
як turn.raw_assistant_message — без змін (там непрозорі поля провайдера).
"""
import json
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from bench import models


@dataclass
class ToolCall:
    id: str
    name: str
    arguments_raw: str
    arguments: dict | None
    parse_error: str | None


@dataclass
class Usage:
    input: int | None = None
    output: int | None = None
    reasoning: int | None = None
    cached: int | None = None


@dataclass
class Turn:
    text: str | None
    parsed: object | None
    schema_valid: bool | None
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: str | None
    latency_ms: int
    model_reported: str | None
    raw_assistant_message: object


OPENAI_PARAMS = {"temperature", "top_p", "seed", "max_tokens", "reasoning_effort", "stop"}
_clients: dict[str, object] = {}


def _client(spec: models.ModelSpec):
    """Фабрика: SDK-клієнт створюється під час першого виклику й кешується."""
    if spec.alias not in _clients:
        key = os.environ.get(spec.key_env or "") or spec.api_key_default
        if not key:
            raise RuntimeError(f"задайте {spec.key_env} у .env")
        if spec.api == "gemini_native":
            _clients[spec.alias] = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=180_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        else:
            _clients[spec.alias] = OpenAI(
                base_url=spec.base_url,
                api_key=key,
                max_retries=0,
                timeout=180.0,
            )
    return _clients[spec.alias]


def to_strict_schema(model: type[BaseModel]) -> dict:
    """JSON Schema для strict: additionalProperties=false і повний required."""
    result = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                fix(value)
        elif isinstance(node, list):
            for value in node:
                fix(value)

    fix(result)
    return result


def _tool_call(call_id: str, name: str, raw: str) -> ToolCall:
    try:
        arguments = json.loads(raw or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments — не JSON-об'єкт")
        return ToolCall(call_id, name, raw, arguments, None)
    except ValueError as exc:
        return ToolCall(call_id, name, raw, None, str(exc))


def _openai_chat(spec, messages, tools, schema, params) -> Turn:
    kwargs = {key: value for key, value in params.items() if key in OPENAI_PARAMS}
    extra = {key: value for key, value in params.items() if key not in OPENAI_PARAMS}
    if spec.alias == "openrouter":
        extra["provider"] = {"require_parameters": True}
    if tools:
        kwargs["tools"] = tools
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": to_strict_schema(schema),
                "strict": True,
            },
        }
    start = time.perf_counter()
    response = _client(spec).chat.completions.create(
        model=spec.model,
        messages=messages,
        extra_body=extra or None,
        **kwargs,
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    choice, usage_raw = response.choices[0], response.usage
    message = choice.message
    usage = Usage(
        input=getattr(usage_raw, "prompt_tokens", None),
        output=getattr(usage_raw, "completion_tokens", None),
        reasoning=getattr(
            getattr(usage_raw, "completion_tokens_details", None), "reasoning_tokens", None
        ),
        cached=getattr(getattr(usage_raw, "prompt_tokens_details", None), "cached_tokens", None),
    )
    calls = [
        _tool_call(call.id, call.function.name, call.function.arguments)
        for call in message.tool_calls or []
    ]
    return Turn(
        message.content,
        None,
        None,
        calls,
        usage,
        choice.finish_reason,
        latency_ms,
        response.model,
        raw_assistant_message=message.model_dump(exclude_none=True),
    )


def _responses_input(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Історія OpenAI-style → параметри Responses API для текстових прогонів."""
    instructions = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
    input_items = []
    for message in messages:
        if message["role"] == "system":
            continue
        if "response_output" in message:
            # A Responses function call must be sent back verbatim before its
            # function_call_output; otherwise the server loses the call_id.
            input_items.extend(message["response_output"])
        if message["role"] == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": message["content"],
            })
        elif "response_output" not in message:
            input_items.append({"role": message["role"], "content": message["content"]})
    return instructions or None, input_items


def _responses_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
            "strict": tool["function"].get("strict", True),
        }
        for tool in tools
    ]


def _responses_chat(spec, messages, tools, schema, params) -> Turn:
    """Один крок через OpenAI Responses API, зведений до спільного Turn."""
    instructions, input_items = _responses_input(messages)
    kwargs = {
        "model": spec.model,
        "input": input_items,
        "instructions": instructions,
        "max_output_tokens": params.get("max_tokens"),
        "reasoning": {"effort": params["reasoning_effort"]} if "reasoning_effort" in params else None,
        "tools": _responses_tools(tools),
    }
    if schema is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": to_strict_schema(schema),
                "strict": True,
            }
        }
    kwargs = {name: value for name, value in kwargs.items() if value is not None}
    start = time.perf_counter()
    response = _client(spec).responses.create(**kwargs)
    latency_ms = round((time.perf_counter() - start) * 1000)
    output = getattr(response, "output", []) or []
    calls = [
        _tool_call(item.call_id, item.name, item.arguments)
        for item in output
        if getattr(item, "type", None) == "function_call"
    ]
    usage_raw = getattr(response, "usage", None)
    usage = Usage(
        input=getattr(usage_raw, "input_tokens", None),
        output=getattr(usage_raw, "output_tokens", None),
        reasoning=getattr(getattr(usage_raw, "output_tokens_details", None), "reasoning_tokens", None),
        cached=getattr(getattr(usage_raw, "input_tokens_details", None), "cached_tokens", None),
    )
    return Turn(
        getattr(response, "output_text", None),
        None,
        None,
        calls,
        usage,
        "tool_calls" if calls else (getattr(response, "status", None) or "stop"),
        latency_ms,
        getattr(response, "model", None),
        raw_assistant_message={"role": "assistant", "response_output": [
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            for item in output
        ]},
    )


def _to_gemini(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Історія OpenAI → (system_instruction, contents) для generate_content."""
    system, contents = None, []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        elif "gemini_content" in message:
            contents.append(message["gemini_content"])
        elif message["role"] == "tool":
            part = types.Part(
                function_response=types.FunctionResponse(
                    id=message["tool_call_id"],
                    name=message["name"],
                    response={"result": message["content"]},
                )
            )
            last = contents[-1] if contents else None
            if last and last.role == "user" and last.parts[0].function_response:
                last.parts.append(part)
            else:
                contents.append(types.Content(role="user", parts=[part]))
        else:
            role = "model" if message["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))
    return system, contents


def _gemini_chat(spec, messages, tools, schema, params) -> Turn:
    system, contents = _to_gemini(messages)
    config = dict(
        system_instruction=system,
        temperature=params.get("temperature"),
        seed=params.get("seed"),
        max_output_tokens=params.get("max_tokens"),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    if "thinking_level" in params:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=params["thinking_level"])
    if schema is not None:
        config.update(
            response_mime_type="application/json",
            response_json_schema=to_strict_schema(schema),
        )
    if tools:
        config["tools"] = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        parameters_json_schema=tool["function"]["parameters"],
                    )
                    for tool in tools
                ]
            )
        ]
    start = time.perf_counter()
    response = _client(spec).models.generate_content(
        model=spec.model,
        contents=contents,
        config=types.GenerateContentConfig(**config),
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    candidate = response.candidates[0] if response.candidates else None
    content = candidate.content if candidate else None
    parts = (content.parts or []) if content else []
    text = "".join(part.text for part in parts if part.text and not part.thought) or None
    calls = [
        ToolCall(
            part.function_call.id or part.function_call.name,
            part.function_call.name,
            json.dumps(part.function_call.args or {}, ensure_ascii=False),
            dict(part.function_call.args or {}),
            None,
        )
        for part in parts
        if part.function_call
    ]
    metadata = response.usage_metadata
    thoughts = metadata.thoughts_token_count if metadata else None
    usage = (
        Usage(
            input=metadata.prompt_token_count,
            output=(metadata.candidates_token_count or 0) + (thoughts or 0),
            reasoning=thoughts,
            cached=metadata.cached_content_token_count,
        )
        if metadata
        else Usage()
    )
    finish = candidate.finish_reason if candidate else None
    finish = str(getattr(finish, "value", finish)).lower() if finish else None
    return Turn(
        text,
        None,
        None,
        calls,
        usage,
        finish,
        latency_ms,
        response.model_version,
        raw_assistant_message={"role": "assistant", "gemini_content": content},
    )


def chat(
    alias: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    schema: type[BaseModel] | None = None,
    **params,
) -> Turn:
    """Один крок розмови. params перекривають spec.params; None не надсилається."""
    spec = models.get(alias)
    merged = {key: value for key, value in {**spec.params, **params}.items() if value is not None}
    if spec.num_ctx is not None:
        merged.setdefault("num_ctx", spec.num_ctx)
    call = {
        "gemini_native": _gemini_chat,
        "openai_responses": _responses_chat,
    }.get(spec.api, _openai_chat)
    turn = call(spec, messages, tools, schema, merged)
    if schema is not None:
        try:
            turn.parsed = schema.model_validate_json(turn.text or "")
            turn.schema_valid = True
        except ValidationError:
            turn.parsed, turn.schema_valid = None, False
    if spec.num_ctx and (turn.usage.input or 0) >= 0.9 * spec.num_ctx:
        print(
            f"УВАГА: {turn.usage.input} вхідних токенів ≥ 90% контексту "
            f"{spec.num_ctx} ({alias})"
        )
    return turn


def generate(
    alias: str,
    system: str,
    user: str,
    schema: type[BaseModel] | None = None,
    **params,
) -> Turn:
    """Тонка обгортка: один запит system + user."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return chat(alias, messages, schema=schema, **params)
