"""Оцінювання: нормалізація, інтервал Вілсона, розбір відповіді й правильність."""
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


def parse_output(text: str) -> object | None:
    """Повертає мітку, лише якщо після norm у тексті є рівно одна допустима мітка."""
    matches = LABEL_PATTERN.findall(norm(text))
    return matches[0] if len(matches) == 1 else None


def is_correct(expected, parsed_or_text) -> bool:
    """Перевіряє збіг розібраної відповіді з expected після norm."""
    return isinstance(parsed_or_text, str) and norm(str(expected)) == norm(parsed_or_text)
