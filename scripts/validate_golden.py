"""Перевірка еталонного набору: uv run python scripts/validate_golden.py [--path ...] [--min N] [--tools ...]"""
import argparse
import codecs
import json
import sys
from collections import Counter

FIELDS = ("id", "input", "expected", "split", "tags")


def check_file(path: str, splits: set[str]) -> tuple[list[dict], list[str]]:
    """Формат кожного рядка; повертає (рядки, помилки з номером рядка)."""
    with open(path, "rb") as file:
        raw = file.read()
    if raw.startswith(codecs.BOM_UTF8):
        return [], [f"{path}: файл має BOM — збережіть як UTF-8 без BOM"]
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        return [], [f"{path}: не UTF-8 ({error})"]
    rows, errors, seen = [], [], set()
    for number, line in enumerate(lines, start=1):
        err = lambda message: errors.append(f"{path}:{number}: {message}")  # noqa: E731
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            err(f"не JSON ({error.msg})")
            continue
        if not isinstance(row, dict) or any(key not in row for key in FIELDS):
            err(f"потрібен JSON-об'єкт з полями {', '.join(FIELDS)}")
            continue
        if not isinstance(row["id"], str) or not row["id"].strip() or row["id"] in seen:
            err(f"id {row['id']!r}: порожній, не рядок або повторюється")
        seen.add(str(row["id"]))
        if not isinstance(row["input"], str) or not row["input"].strip():
            err("input — непорожній рядок")
        expected = row["expected"]
        if not isinstance(expected, (str, dict, list)) or not (expected.strip() if isinstance(expected, str) else expected):
            err("expected — непорожній рядок, об'єкт або список")
        if not isinstance(row["split"], str) or row["split"] not in splits:
            err(f"split має бути одним із {sorted(splits)}")
        if not isinstance(row["tags"], list) or not all(isinstance(tag, str) for tag in row["tags"]):
            err("tags — список рядків")
            row["tags"] = []
        rows.append(row)
    return rows, errors


def summary(rows: list[dict]) -> str:
    return (
        f"рядків: {len(rows)} · split: {dict(Counter(str(row['split']) for row in rows))} · "
        f"теги: {dict(Counter(tag for row in rows for tag in row['tags']))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/golden.jsonl")
    parser.add_argument("--min", type=int, help="чернетка: лише формат і щонайменше N рядків")
    parser.add_argument("--tools", help="файл tool-кейсів (split = tool)")
    args = parser.parse_args()
    rows, errors = check_file(args.path, {"dev", "test"})
    print(summary(rows))
    if args.min is not None:
        errors += [f"рядків {len(rows)}, потрібно ≥{args.min}"] if len(rows) < args.min else []
    else:
        dev, test = (sum(row["split"] == split for row in rows) for split in ("dev", "test"))
        tagged = {tag: [row for row in rows if tag in row["tags"]] for tag in ("hard", "injection", "q15")}
        checks = [
            (len(rows) >= 33, f"рядків {len(rows)}, потрібно ≥33"),
            (dev == 8, f"dev = {dev}, потрібно рівно 8"),
            (test >= 25, f"test = {test}, потрібно ≥25"),
            (len(tagged["hard"]) >= 5, f"hard = {len(tagged['hard'])}, потрібно ≥5"),
            (len(tagged["injection"]) >= 2, f"injection = {len(tagged['injection'])}, потрібно ≥2"),
            (len(tagged["q15"]) == 15, f"q15 = {len(tagged['q15'])}, потрібно рівно 15"),
        ]
        errors += [message for ok, message in checks if not ok]
        errors += [
            f"{row['id']}: тег {tag} дозволений лише в test"
            for tag, tagged_rows in tagged.items()
            for row in tagged_rows
            if row["split"] != "test"
        ]
    if args.tools:
        tool_rows, tool_errors = check_file(args.tools, {"tool"})
        print(summary(tool_rows))
        errors += tool_errors + ([f"{args.tools}: рядків {len(tool_rows)}, потрібно ≥8"] if len(tool_rows) < 8 else [])
        overlap = {str(row["id"]) for row in rows} & {str(row["id"]) for row in tool_rows}
        errors += [f"{args.tools}: id перетинаються з golden: {sorted(overlap)}"] if overlap else []
    for error in errors:
        print("ПОМИЛКА", error)
    print("OK" if not errors else f"помилок: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
