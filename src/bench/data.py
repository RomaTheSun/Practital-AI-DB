"""Дані: приклади з JSONL, вибірки, хеші й рядки результатів."""
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Example:
    id: str
    input: str
    expected: object
    split: str
    tags: list[str]


def load_jsonl(path: str | Path) -> list[Example]:
    """Рядки JSONL (UTF-8) → список Example; порожні рядки пропускаються."""
    with open(path, encoding="utf-8") as file:
        return [Example(**json.loads(line)) for line in file if line.strip()]


def select(
    examples: list[Example],
    split: str | None = None,
    tag: str | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
) -> list[Example]:
    """Фільтрує приклади, зберігаючи порядок у файлі."""
    selected = [
        example
        for example in examples
        if (split is None or example.split == split)
        and (tag is None or tag in example.tags)
        and (ids is None or example.id in ids)
    ]
    return selected[:limit] if limit is not None else selected


def sha256_file(path: str | Path) -> str:
    """Повертає SHA-256 байтів файлу в hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FIELDS = (
    "run_id", "ts", "git_sha", "dataset_sha256", "prompt_version", "prompt_sha256",
    "alias", "api", "model_requested", "model_reported", "params", "mode", "id", "split", "tags",
    "output_raw", "parsed", "schema_valid", "correct", "finish_reason",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
    "latency_ms", "attempt", "concurrency", "error_type", "error",
)


def new_row(**fields) -> dict:
    """Створює рядок з усіма обов'язковими полями."""
    return {**dict.fromkeys(FIELDS), "tags": [], **fields}


def read_rows(path: str | Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_row(path: str | Path, row: dict) -> None:
    """Лише дописує рядок у кінець файлу."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str | None:
    """Короткий SHA поточного коміту або None."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
