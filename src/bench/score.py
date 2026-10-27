"""Оцінювання: нормалізація, інтервал Вілсона, розбір відповіді й правильність."""
import json
import math
import re
import unicodedata

LABELS = ("postgresql", "mongodb", "redis", "neo4j")
LABEL_PATTERN = re.compile(r"(?<!\w)(postgresql|mongodb|redis|neo4j)(?!\w)")


def norm(s: str) -> str:
    """NFC, уніфікація апострофів, стискання пробілів і casefold."""
    s = unicodedata.normalize("NFC", s).replace("’", "'").replace("ʼ", "'")
    return " ".join(s.split()).casefold()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% інтервал Вілсона для частки k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, center - half), min(1.0, center + half))


def clean(text: str) -> str:
    """Прибирає приховане міркування і суцільну Markdown-огорожу JSON."""
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, flags=re.S | re.I)
    return fence.group(1).strip() if fence else body


def _valid_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = norm(value)
    return value if value in LABELS else None


def parse_output(text: str) -> object | None:
    """Мітка з JSON, однозначного жирного виділення або однозначного тексту."""
    body = clean(text)
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        decoded = None
    if isinstance(decoded, dict):
        label = _valid_label(decoded.get("label"))
        if label is not None:
            return label

    normalized = norm(body)
    bold = {
        label
        for label in LABELS
        if re.search(rf"\*\*\s*{re.escape(label)}\s*\*\*", normalized)
    }
    if bold:
        return next(iter(bold)) if len(bold) == 1 else None

    found = set(LABEL_PATTERN.findall(normalized))
    return next(iter(found)) if len(found) == 1 else None


def is_correct(expected, parsed_or_text) -> bool:
    """Порівнює expected з міткою у словнику, сирому тексті або вже розібраній мітці."""
    if isinstance(parsed_or_text, dict):
        got = _valid_label(parsed_or_text.get("label"))
    elif isinstance(parsed_or_text, str):
        direct = _valid_label(parsed_or_text)
        got = direct if direct is not None else parse_output(parsed_or_text)
    else:
        got = None
    return isinstance(got, str) and norm(str(expected)) == norm(got)
