"""Крок 02 · множник токенів ua/en."""
import argparse
import json
from datetime import datetime, timezone

import tiktoken
from dotenv import load_dotenv

from bench import clients, data, models


def model_counter(alias: str):
    """Лічильник токенів локальної моделі з відніманням chat-базової лінії."""
    def usage_input(text: str) -> int:
        usage = clients.chat(alias, [{"role": "user", "content": text}], max_tokens=1)["usage"]["input"]
        if usage is None:
            raise RuntimeError("сервер не повернув prompt_tokens")
        return usage

    baseline = usage_input("")
    return lambda text: usage_input(text) - baseline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="local")
    parser.add_argument("--pairs", default="data/ua_en.jsonl")
    parser.add_argument("--out", default="results/02-tokens.jsonl")
    args = parser.parse_args()
    spec = models.get(args.model)
    with open(args.pairs, encoding="utf-8") as file:
        pairs = [json.loads(line) for line in file if line.strip()]
    encoding = tiktoken.get_encoding("o200k_base")
    counters = {
        "o200k_base": lambda text: len(encoding.encode(text)),
        f"{spec.alias}:{spec.model}": model_counter(spec.alias),
    }
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%MZ}_tokens_{spec.alias}"
    dataset_sha = data.sha256_file(args.pairs)
    params = {**spec.params, "num_ctx": spec.num_ctx, "max_tokens": 1}
    for pair in pairs:
        for tokenizer, count in counters.items():
            ua_tokens, en_tokens = count(pair["ua"]), count(pair["en"])
            if en_tokens <= 0:
                raise RuntimeError(f"{pair['id']}: англійський текст дав {en_tokens} токенів")
            data.append_row(
                args.out,
                data.new_row(
                    run_id=run_id,
                    ts=data.utc_now(),
                    git_sha=data.git_sha(),
                    dataset_sha256=dataset_sha,
                    alias=spec.alias,
                    api=spec.api,
                    model_requested=spec.model,
                    params=params,
                    mode="tokens",
                    id=pair["id"],
                    tokenizer=tokenizer,
                    ua_tokens=ua_tokens,
                    en_tokens=en_tokens,
                    ratio=round(ua_tokens / en_tokens, 3),
                ),
            )
            print(f"{pair['id']} · {tokenizer}: {ua_tokens} / {en_tokens} = {ua_tokens / en_tokens:.2f}")


if __name__ == "__main__":
    main()
