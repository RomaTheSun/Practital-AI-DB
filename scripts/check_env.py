"""Перевірка середовища (крок 00): uv run python scripts/check_env.py [--out docs/lab1/env.md] [--live]"""
import argparse
import ctypes
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from importlib import metadata
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SDKS = ("openai", "google-genai", "pydantic", "tiktoken", "python-dotenv")


def http_json(url: str, payload: dict | None = None, key: str | None = None, timeout: float = 3.0) -> dict:
    """GET (payload=None) або POST JSON через urllib — без SDK."""
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def safely(check) -> tuple[bool, str]:
    """Мережева перевірка -> (OK?, деталі) без трейсбека; текст ключа не друкується."""
    try:
        return check()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (OSError, ValueError, KeyError) as e:
        return False, f"помилка {type(e).__name__}"


def local_model(base_url: str, model: str) -> tuple[bool, str]:
    ids = [m["id"] for m in http_json(f"{base_url}/models")["data"]]
    return model in ids, f"{model}: {'є' if model in ids else 'немає'} серед {len(ids)}"


def canary(base_url: str, model: str) -> tuple[bool, str]:
    """~6000 токенів службового тексту: контрольне слово на початку, прохання повторити — в кінці."""
    word = f"КАНАРКА-{random.randint(1000, 9999)}"
    filler = "\n".join(f"Рядок {i}: службовий текст для перевірки довжини контексту." for i in range(1, 281))
    prompt = f"Контрольне слово: {word}.\n\n{filler}\n\nЯке контрольне слово було на самому початку? Відповідай лише ним."
    response = http_json(
        f"{base_url}/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 512,
            "reasoning_effort": "none",
        },
        timeout=600,
    )
    text = response["choices"][0]["message"].get("content") or ""
    return word in text, f"prompt_tokens={response.get('usage', {}).get('prompt_tokens')} · відповідь: {text.strip()[:30]}"


def cloud_call() -> tuple[bool, str]:
    """Один крихітний хмарний виклик (Gemini або OpenRouter) через OpenAI-сумісну адресу."""
    env = os.environ.get
    if env("GEMINI_API_KEY"):
        url, key, model = (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            env("GEMINI_API_KEY"),
            "gemini-3.8-flash",
        )
    elif env("OPENROUTER_API_KEY") and env("OPENROUTER_MODEL"):
        url, key, model = "https://openrouter.ai/api/v1/chat/completions", env("OPENROUTER_API_KEY"), env("OPENROUTER_MODEL")
    else:
        return False, "немає хмарного ключа"
    response = http_json(url, {"model": model, "messages": [{"role": "user", "content": "Скажи одне слово: так"}]}, key, 60)
    return True, f"{model}: {(response['choices'][0]['message'].get('content') or '').strip()[:20]}"


def ram() -> str:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong)] + [
                    (f"x{i}", ctypes.c_ulonglong) for i in range(6)
                ]

            stat = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total = stat.total
        else:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return f"{total / 2**30:.1f} ГБ"
    except (AttributeError, OSError, ValueError):
        return "невідомо"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Markdown-звіт, напр. docs/lab1/env.md")
    parser.add_argument("--live", action="store_true", help="канарка контексту і хмарний виклик")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    base_url = (env("LOCAL_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
    model = env("LOCAL_MODEL") or "qwen3:4b"
    engine = {11434: "Ollama", 1234: "LM Studio"}.get(urlparse(base_url).port, "невідомо")
    ctx = {key: int(env(key)) if (env(key) or "").isdigit() else 0 for key in ("LOCAL_NUM_CTX", "OLLAMA_CONTEXT_LENGTH")}
    rows: list[tuple[str, str, str, bool]] = []

    def add(name: str, ok: bool | None, detail: str, required: bool = False) -> None:
        rows.append((name, "OK" if ok else "—" if ok is None else "FIX", detail, required))

    git = shutil.which("git")
    add("Python ≥ 3.13", sys.version_info >= (3, 13), platform.python_version(), True)
    add(
        "git у PATH",
        bool(git),
        subprocess.run([git, "--version"], capture_output=True, text=True).stdout.strip() if git else "не знайдено",
        True,
    )
    try:
        import bench  # noqa: F401

        add("import bench", True, "імпортується", True)
    except ImportError as error:
        add("import bench", False, str(error), True)
    add(".env існує", (ROOT / ".env").exists(), "поруч із pyproject.toml; шаблон — .env.example", True)
    cloud = env("GEMINI_API_KEY") or (env("OPENROUTER_API_KEY") and env("OPENROUTER_MODEL"))
    add(
        "хмарний ключ",
        bool(cloud),
        " · ".join(
            f"{key}: {'задано' if env(key) else 'не задано'}"
            for key in ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_MODEL")
        ),
        True,
    )
    ok, detail = safely(lambda: local_model(base_url, model))
    add("локальний сервер і модель", ok, f"{engine} ({base_url}) · {detail}")
    add(
        "контекст ≥ 16384",
        ctx["LOCAL_NUM_CTX"] >= 16384 and (engine != "Ollama" or ctx["OLLAMA_CONTEXT_LENGTH"] >= 16384),
        " · ".join(f"{key}={value or 'не задано'}" for key, value in ctx.items()),
    )
    for name, check in (("канарка контексту", lambda: canary(base_url, model)), ("хмарний виклик", cloud_call)):
        if args.live:
            add(name, *safely(check))
        else:
            add(name, None, "запустіть з --live")

    table = ["| # | Перевірка | Стан | Деталі |", "|---|---|---|---|"]
    table += [f"| {i} | {name} | {state} | {detail} |" for i, (name, state, detail, _) in enumerate(rows, 1)]
    print("\n".join(table))
    if args.out:
        sdk = " · ".join(f"{package} {next((dist.version for dist in metadata.distributions(name=package)), '—')}" for package in SDKS)
        head = [
            "# Середовище стенду",
            "",
            f"- Дата: {date.today():%Y-%m-%d}",
            f"- ОС: {platform.platform()}",
            f"- Python: {platform.python_version()}",
            f"- RAM: {ram()}",
            f"- Локальний рушій: {engine} ({base_url})",
            f"- Модель: {model}",
            f"- Контекст: {ctx['LOCAL_NUM_CTX'] or 'не задано'}",
            f"- SDK: {sdk}",
            "",
        ]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as file:
            file.write("\n".join(head + table) + "\n")
        print(f"записано {args.out}")
    return 1 if any(state == "FIX" and required for _, state, _, required in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
