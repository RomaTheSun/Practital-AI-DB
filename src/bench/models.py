"""Реєстр моделей v1 (крок 03): умови відтворюваного експерименту."""
import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    api: str
    model: str
    base_url: str | None
    key_env: str | None
    api_key_default: str | None = None
    num_ctx: int | None = None
    params: dict = field(default_factory=dict)
    caps: dict = field(default_factory=dict)
    price_in: float = 0.0
    price_out: float = 0.0
    price_src: str = ""
    price_date: str = ""
    free_tier: bool = False
    trains_on_data: bool | None = None


GEMINI_PRICE = dict(
    price_in=0.75,
    price_out=3.75,
    price_src="https://ai.google.dev/gemini-api/docs/pricing",
    price_date="2026-10-10",
)

OPENAI_LUNA_PRICE = dict(
    price_in=0.10,
    price_out=0.50,
    price_src="https://developers.openai.com/api/docs/models/gpt-6-luna",
    price_date="2026-11-07",
)


MODELS: dict[str, ModelSpec] = {
    "local": ModelSpec(
        alias="local",
        api="openai_chat",
        model="qwen3:4b",
        base_url="http://localhost:11434/v1",
        key_env=None,
        api_key_default="ollama",
        num_ctx=16384,
        params={"temperature": 0, "seed": 7, "reasoning_effort": "high", "max_tokens": 4096},
        caps={"tools": True, "json_schema": True, "reasoning_control": True},
        price_src="локальна модель",
        price_date="2026-10-27",
        free_tier=True,
        trains_on_data=False,
    ),
    "openai": ModelSpec(
        alias="openai",
        api="openai_chat",
        model="gpt-6-luna",
        base_url=None,
        key_env="OPENAI_API_KEY",
        params={"reasoning_effort": "none"},
        caps={"tools": True, "json_schema": True, "reasoning_control": True},
        free_tier=False,
        trains_on_data=None,
        **OPENAI_LUNA_PRICE,
    ),
    "openai-responses": ModelSpec(
        alias="openai-responses",
        api="openai_responses",
        model="gpt-6-luna",
        base_url=None,
        key_env="OPENAI_API_KEY",
        params={"reasoning_effort": "low"},
        caps={"tools": True, "json_schema": True, "reasoning_control": True},
        free_tier=False,
        trains_on_data=None,
        **OPENAI_LUNA_PRICE,
    ),
    "gemini": ModelSpec(
        alias="gemini",
        api="openai_chat",
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        params={"reasoning_effort": "low"},
        caps={"tools": True, "json_schema": True, "reasoning_control": True},
        free_tier=True,
        trains_on_data=True,
        **GEMINI_PRICE,
    ),
    "gemini-native": ModelSpec(
        alias="gemini-native",
        api="gemini_native",
        model="gemini-3.8-flash",
        base_url=None,
        key_env="GEMINI_API_KEY",
        params={"thinking_level": "low"},
        caps={"tools": True, "json_schema": True, "reasoning_control": True},
        free_tier=True,
        trains_on_data=True,
        **GEMINI_PRICE,
    ),
}


def get(alias: str) -> ModelSpec:
    """Повертає специфікацію з урахуванням змінних середовища (.env)."""
    if alias not in MODELS:
        raise KeyError(f"невідомий псевдонім {alias!r}; доступні: {', '.join(MODELS)}")
    spec = MODELS[alias]
    env = os.environ.get
    if alias == "local":
        spec = replace(
            spec,
            model=env("LOCAL_MODEL") or spec.model,
            base_url=env("LOCAL_BASE_URL") or spec.base_url,
            num_ctx=int(env("LOCAL_NUM_CTX") or spec.num_ctx),
        )
    return spec
