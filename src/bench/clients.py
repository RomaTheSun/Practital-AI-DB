"""Клієнт LLM v1 (крок 02): chat() над OpenAI-сумісною формою API."""
import os
import time

from openai import OpenAI

from bench import models

STANDARD = {"temperature", "top_p", "seed", "max_tokens", "reasoning_effort", "stop"}
_clients: dict[str, OpenAI] = {}


def _client(spec: models.ModelSpec) -> OpenAI:
    """Створює й кешує клієнт під час першого виклику."""
    if spec.alias not in _clients:
        key = os.environ.get(spec.key_env or "") or spec.api_key_default
        if not key:
            raise RuntimeError(f"задайте {spec.key_env} у .env")
        _clients[spec.alias] = OpenAI(base_url=spec.base_url, api_key=key)
    return _clients[spec.alias]


def chat(alias: str, messages: list[dict], **params) -> dict:
    """Виконує один запит; params перекривають реєстр, None означає «не надсилати»."""
    spec = models.get(alias)
    if spec.api != "openai_chat":
        raise ValueError(f"clients v1 підтримує лише openai_chat, а {alias} — {spec.api}")
    defaults = {**spec.params}
    if spec.num_ctx is not None:
        defaults["num_ctx"] = spec.num_ctx
    merged = {key: value for key, value in {**defaults, **params}.items() if value is not None}
    standard = {key: value for key, value in merged.items() if key in STANDARD}
    extra = {key: value for key, value in merged.items() if key not in STANDARD}
    start = time.perf_counter()
    response = _client(spec).chat.completions.create(
        model=spec.model,
        messages=messages,
        extra_body=extra or None,
        **standard,
    )
    latency_ms = round((time.perf_counter() - start) * 1000)
    choice, usage = response.choices[0], response.usage
    return {
        "text": choice.message.content,
        "usage": {
            "input": getattr(usage, "prompt_tokens", None),
            "output": getattr(usage, "completion_tokens", None),
            "reasoning": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
            "cached": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None),
        },
        "finish_reason": choice.finish_reason,
        "latency_ms": latency_ms,
        "model_reported": response.model,
        "raw": response.model_dump(),
    }
