import json
from pathlib import Path

from bench import clients, run

PROMPT = "version: v0\n\n## system\nТест.\n\n## user\n<<input>>\n"


def make_files(tmp_path: Path) -> tuple[str, str]:
    rows = [{"id": f"t{i:02d}", "input": f"вхід {i}", "expected": "postgresql", "split": "test", "tags": []} for i in range(1, 5)]
    data_path, prompt_path = tmp_path / "golden.jsonl", tmp_path / "v0.md"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    prompt_path.write_text(PROMPT, encoding="utf-8")
    return str(data_path), str(prompt_path)


def fake_chat(alias, messages, **params):
    return {"text": '{"label": "postgresql"}', "finish_reason": "stop", "latency_ms": 12,
            "usage": {"input": 100, "output": 5, "reasoning": None, "cached": None},
            "model_reported": "fake-model", "raw": {}}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_resume_only_appends(tmp_path, monkeypatch):
    data_path, prompt_path = make_files(tmp_path)
    monkeypatch.setattr(clients, "chat", fake_chat)
    out = tmp_path / "03.jsonl"
    args = ["--prompt", prompt_path, "--model", "local", "--data", data_path, "--split", "test", "--out", str(out)]
    run.main(args + ["--limit", "2"])
    before = out.read_bytes()
    run.main(args + ["--limit", "4", "--resume"])
    assert out.read_bytes().startswith(before)
    assert [row["id"] for row in rows(out)] == ["t01", "t02", "t03", "t04"]
    assert rows(out)[0]["correct"] is True and rows(out)[0]["attempt"] == 1


def test_rate_limit_becomes_error_row_then_resume(tmp_path, monkeypatch):
    class RateLimited(Exception):
        status_code = 429

    def failing(alias, messages, **params):
        raise RateLimited("429 Too Many Requests")

    data_path, prompt_path = make_files(tmp_path)
    monkeypatch.setattr(run.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(clients, "chat", failing)
    out = tmp_path / "03.jsonl"
    args = ["--prompt", prompt_path, "--model", "local", "--data", data_path, "--ids", "t03", "--out", str(out)]
    run.main(args)
    assert rows(out)[0]["error_type"] == "rate_limit" and rows(out)[0]["correct"] is None
    monkeypatch.setattr(clients, "chat", fake_chat)
    run.main(args + ["--resume"])
    second = rows(out)[1]
    assert second["id"] == "t03" and second["attempt"] == 2 and second["error_type"] is None
