"""Реєстр моделей v1 (крок 02): псевдонім → форма API, ID моделі, ключ і параметри."""
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


MODELS: dict[str, ModelSpec] = {
    "local": ModelSpec(
        alias="local",
        api="openai_chat",
        model="qwen3:4b",
        base_url="http://localhost:11434/v1",
        key_env=None,
        api_key_default="ollama",
        num_ctx=16384,
        params={"temperature": 0, "seed": 7, "reasoning_effort": "none"},
    ),
    "gemini": ModelSpec(
        alias="gemini",
        api="openai_chat",
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        params={"reasoning_effort": "low"},
    ),
    "openrouter": ModelSpec(
        alias="openrouter",
        api="openai_chat",
        model="",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
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
    elif alias == "openrouter":
        spec = replace(spec, model=env("OPENROUTER_MODEL") or spec.model)
    return spec
