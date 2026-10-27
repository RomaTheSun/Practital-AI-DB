"""Прогін набору через модель: uv run python -m bench.run --prompt prompts/v1.md --model local."""
import argparse
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from bench import clients, data, models, prompts, score

RETRYABLE = {"rate_limit", "server", "timeout"}


def error_type(exc: Exception) -> str:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 429:
        return "rate_limit"
    if isinstance(code, int) and code >= 500:
        return "server"
    if "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "schema" if code == 400 and "schema" in str(exc).lower() else "other"


def with_retries(call, attempts: int = 5):
    """Повторює 429/5xx/timeout; повертає (результат, помилка)."""
    for index in range(1, attempts + 1):
        try:
            return call(), None
        except Exception as exc:  # noqa: BLE001
            kind = error_type(exc)
            if kind not in RETRYABLE or index == attempts:
                message = f"{type(exc).__name__}: {exc}"[:500]
                for name, value in os.environ.items():
                    if name.endswith("_API_KEY") and value:
                        message = message.replace(value, "***")
                return None, (kind, message)
            time.sleep(min(60, 2**index) + random.uniform(0, 1))


def evaluate(example: data.Example, text: str | None) -> tuple[object, bool]:
    parsed = score.parse_output(text or "")
    return parsed, parsed is not None and score.is_correct(example.expected, parsed)


def fill_turn(row: dict, example: data.Example, response: dict, mode: str | None) -> None:
    parsed, correct = evaluate(example, response["text"])
    usage = response["usage"]
    row.update(
        model_reported=response["model_reported"],
        output_raw=response["text"],
        parsed=parsed,
        schema_valid=None,
        correct=correct,
        finish_reason=response["finish_reason"],
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        reasoning_tokens=usage["reasoning"],
        cached_tokens=usage["cached"],
        latency_ms=response["latency_ms"],
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогін набору через модель")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True, help="псевдонім з models.py")
    parser.add_argument("--data", default="data/golden.jsonl")
    parser.add_argument("--split")
    parser.add_argument("--tag")
    parser.add_argument("--ids", help="id через кому")
    parser.add_argument("--ids-file", help="файл з id, по одному в рядку")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=["A", "B"], help="зарезервовано для кроку 04")
    parser.add_argument("--tools", help="зарезервовано для кроку 05")
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rpm", type=float)
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def default_out(args: argparse.Namespace, version: str) -> str:
    return f"results/03-{version}-{args.model}.jsonl"


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    if args.mode == "B" or args.tools is not None:
        raise SystemExit("режими схеми та інструментів з'являться на кроках 04–05")
    spec, prompt = models.get(args.model), prompts.load_prompt(args.prompt)
    examples = data.load_jsonl(args.data)
    ids = args.ids.split(",") if args.ids else None
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as file:
            ids = file.read().split()
    out = Path(args.out or default_out(args, prompt.version))
    previous = data.read_rows(out)
    done = {row["id"] for row in previous if row["error_type"] is None}
    attempts = Counter(row["id"] for row in previous)
    selected = data.select(examples, split=args.split, tag=args.tag, ids=ids, limit=args.limit)
    todo = [example for example in selected if not (args.resume and example.id in done)]
    dev = data.select(examples, split="dev")
    run_id = args.run_id or f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%MZ}_{prompt.version}_{args.model}"
    common = dict(
        run_id=run_id,
        git_sha=data.git_sha(),
        dataset_sha256=data.sha256_file(args.data),
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
        alias=spec.alias,
        api=spec.api,
        model_requested=spec.model,
        params={**spec.params, "num_ctx": spec.num_ctx},
        mode=args.mode or "A",
        concurrency=1,
    )
    print(f"{run_id}: {len(todo)} прикладів → {out}")
    for example in todo:
        messages = prompts.render(prompt, example, dev)
        row = data.new_row(
            **common,
            id=example.id,
            split=example.split,
            tags=example.tags,
            attempt=attempts[example.id] + 1,
        )
        result, error = with_retries(lambda: clients.chat(spec.alias, messages))
        row["ts"] = data.utc_now()
        if error:
            row.update(error_type=error[0], error=error[1])
        else:
            fill_turn(row, example, result, args.mode)
        data.append_row(out, row)
        attempts[example.id] += 1
        print(f"  {example.id}: {row['error_type'] or ('правильно' if row['correct'] else 'хибно')}")
        if args.max_rpm:
            time.sleep(60 / args.max_rpm)


if __name__ == "__main__":
    main()
