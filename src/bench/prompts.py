"""Промпти: Markdown-файл з рядком «version: …» і розділами system/user."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from bench.data import Example, sha256_file


@dataclass
class Prompt:
    version: str
    sha256: str
    system: str
    user: str


def load_prompt(path: str | Path) -> Prompt:
    """Читає версію, секції system/user і SHA-256 усього файла."""
    text = Path(path).read_text(encoding="utf-8")
    version_match = re.search(r"(?m)^version:\s*([^\s]+)\s*$", text)
    system_match = re.search(r"(?ms)^## system\s*\n(.*?)(?=^## user\s*$)", text)
    user_match = re.search(r"(?ms)^## user\s*\n(.*)\Z", text)
    missing = [
        name
        for name, match in (("version", version_match), ("## system", system_match), ("## user", user_match))
        if match is None
    ]
    if missing:
        raise ValueError(f"{path}: відсутні або неправильно оформлені: {', '.join(missing)}")
    return Prompt(
        version=version_match.group(1),
        sha256=sha256_file(path),
        system=system_match.group(1).strip(),
        user=user_match.group(1).strip(),
    )


def format_examples(examples: list[Example]) -> str:
    """Форматує few-shot тільки з dev, у JSON-форматі відповіді промпту."""
    non_dev = [example.id for example in examples if example.split != "dev"]
    if non_dev:
        raise ValueError(f"few-shot дозволено лише з dev; отримано: {', '.join(non_dev)}")
    blocks = []
    for example in examples:
        answer = json.dumps({"label": example.expected}, ensure_ascii=False)
        blocks.append(f"Вхід:\n{example.input}\nВідповідь:\n{answer}")
    return "\n\n".join(blocks)


def render(prompt: Prompt, example: Example, dev_examples: list[Example]) -> list[dict]:
    """Підставляє вхід та dev-приклади й екранує закриваючий тег розмежування."""
    input_text = example.input
    wrapper = re.search(r"<([A-Za-z0-9_]+)>\s*<<input>>\s*</\1>", prompt.user, flags=re.I)
    if wrapper:
        tag = wrapper.group(1)
        input_text = re.sub(
            rf"</{re.escape(tag)}>",
            f"&lt;/{tag}&gt;",
            input_text,
            flags=re.I,
        )
    examples_text = format_examples(dev_examples) if "<<examples>>" in prompt.user else ""
    user = prompt.user.replace("<<input>>", input_text).replace("<<examples>>", examples_text)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user},
    ]
