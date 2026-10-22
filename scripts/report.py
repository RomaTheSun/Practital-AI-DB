"""Генерує стабільні Markdown-таблиці кроку 02 з results/*.jsonl."""
import argparse
import glob
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bench import models
from bench.score import norm, wilson_ci

NO_DATA = "_Немає результатів — запустіть прогін._"


def write_block(md_path: str, name: str, markdown: str) -> None:
    with open(md_path, encoding="utf-8", newline="") as file:
        text = file.read()
    pattern = re.compile(rf"(<!-- report:begin {re.escape(name)} -->)(.*?)(<!-- report:end -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"{md_path}: немає блоку report:begin {name} … report:end")
    new = pattern.sub(lambda match: f"{match.group(1)}\n{markdown.strip()}\n{match.group(3)}", text, count=1)
    if new != text:
        with open(md_path, "w", encoding="utf-8", newline="") as file:
            file.write(new)


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as file:
            rows += [{**json.loads(line), "_file": Path(path).name} for line in file if line.strip()]
    return rows


def by_file(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["_file"]].append(row)
    for name, group in groups.items():
        for field in ("prompt_sha256", "dataset_sha256"):
            if len({row[field] for row in group}) > 1:
                raise SystemExit(f"{name}: різні {field} в одному файлі")
    return dict(sorted(groups.items()))


def latest_ok(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows if row.get("error_type") is None}


def error_summary(rows: list[dict]) -> str:
    ok, last = latest_ok(rows), {row["id"]: row for row in rows}
    kinds = Counter(row["error_type"] for identifier, row in last.items() if identifier not in ok)
    return ", ".join(f"{kind} {count}" for kind, count in sorted(kinds.items())) or "0"


def cost(rows: list[dict], prices: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for row in rows:
        spec = models.MODELS.get(row["alias"])
        price_in, price_out = prices.get(
            row["alias"], (spec.price_in, spec.price_out) if spec else (0.0, 0.0)
        )
        total += ((row.get("input_tokens") or 0) * price_in + (row.get("output_tokens") or 0) * price_out) / 1e6
    return total


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows])


def frac(k: int, n: int) -> str:
    return f"{k}/{n}" if n else "—"


def acc_table(groups: dict[str, list[dict]], prices: dict) -> str:
    table = []
    for name, rows in groups.items():
        ok = list(latest_ok(rows).values())
        k, n = sum(bool(row["correct"]) for row in ok), len(ok)
        low, high = wilson_ci(k, n)
        tagged = lambda tag: [row for row in ok if tag in row["tags"]]  # noqa: E731
        table.append([
            name,
            rows[0]["model_requested"],
            rows[0]["prompt_version"],
            n,
            f"{k / n:.2f}" if n else "—",
            f"[{low:.2f}; {high:.2f}]",
            frac(sum(row["correct"] for row in tagged("hard")), len(tagged("hard"))),
            frac(sum(row["correct"] for row in tagged("injection")), len(tagged("injection"))),
            error_summary(rows),
            f"{cost(ok, prices):.4f}",
        ])
    return md_table(
        ["файл", "модель", "промпт", "n", "точність", "95% ДІ", "hard", "injection", "помилки", "вартість, $"],
        table,
    )


def last_run(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["run_id"] == rows[-1]["run_id"]] if rows else []


def block_tokens(_prices: dict) -> str:
    ratios = defaultdict(dict)
    for row in load_rows("results/02-tokens.jsonl"):
        ratios[row["tokenizer"]][row["id"]] = row["ratio"]
    table = []
    for tokenizer, by_id in sorted(ratios.items()):
        values = list(by_id.values())
        quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else values * 3
        table.append([tokenizer, len(values), f"{statistics.median(values):.2f}", f"{quartiles[0]:.2f}–{quartiles[2]:.2f}"])
    return md_table(["токенізатор", "пар", "медіана ua/en", "міжквартильний розмах"], table) if table else NO_DATA


def block_sampling(_prices: dict) -> str:
    rows = last_run(load_rows("results/02-sampling.jsonl"))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["params"].get("temperature"), row["params"].get("seed"))].append(row)
    table = []
    for (temperature, seed), group in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        table.append([
            temperature,
            seed if seed is not None else "не задано",
            len(group),
            len({norm(row["output_raw"] or "") for row in group}),
            sum(bool(row["correct"]) for row in group),
        ])
    return md_table(["temperature", "seed", "прогонів", "різних відповідей", "правильних"], table) if table else NO_DATA


def block_reasoning(_prices: dict) -> str:
    rows = last_run(load_rows("results/02-reasoning.jsonl"))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_requested"], row["params"].get("reasoning_effort"))].append(row)
    table = []
    for (model, level), group in sorted(groups.items()):
        output = [row["output_tokens"] or 0 for row in group]
        chars = [row.get("reasoning_chars") or 0 for row in group]
        latency = [row["latency_ms"] or 0 for row in group]
        finishes = Counter(row["finish_reason"] for row in group)
        finish_text = ", ".join(f"{key}: {value}" for key, value in sorted(finishes.items()))
        table.append([
            model,
            level,
            frac(sum(bool(row["correct"]) for row in group), len(group)),
            f"{statistics.mean(output):.1f}",
            f"{statistics.mean(chars):.1f}",
            f"{statistics.median(latency):.0f}",
            finish_text,
        ])
    return md_table(
        ["модель", "рівень", "правильних", "вихідних токенів, сер.", "символів міркування, сер.", "затримка, мс (медіана)", "finish_reason"],
        table,
    ) if table else NO_DATA


BLOCKS = {
    "tokens": ("docs/lab1/basics.md", block_tokens),
    "sampling": ("docs/lab1/basics.md", block_sampling),
    "reasoning": ("docs/lab1/basics.md", block_reasoning),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("blocks", nargs="*", help=f"блоки: {', '.join(BLOCKS)}")
    parser.add_argument("--price", action="append", default=[], help="alias=IN/OUT")
    args = parser.parse_args()
    unknown = set(args.blocks) - set(BLOCKS)
    if unknown:
        raise SystemExit(f"невідомі блоки: {sorted(unknown)}")
    prices = {}
    for item in args.price:
        alias, pair = item.split("=")
        price_in, price_out = pair.split("/")
        prices[alias] = (float(price_in), float(price_out))
    for name in args.blocks or BLOCKS:
        md_path, build = BLOCKS[name]
        write_block(md_path, name, build(prices))
        print(f"{md_path}: {name}")


if __name__ == "__main__":
    main()
