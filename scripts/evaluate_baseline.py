"""Обчислює accuracy baseline на test у data/golden.jsonl."""
import json
from pathlib import Path

from bench.baseline import predict


def main() -> None:
    rows = [json.loads(line) for line in Path("data/golden.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [row for row in rows if row["split"] == "test"]
    correct = sum(predict(row["input"]) == row["expected"] for row in test)
    print(f"baseline=majority(postgresql) · correct={correct} · n={len(test)} · accuracy={correct / len(test):.3f}")


if __name__ == "__main__":
    main()
