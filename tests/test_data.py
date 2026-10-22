import json

from bench import data


def test_load_select_and_hash(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "a", "input": "A", "expected": "redis", "split": "dev", "tags": []}),
            "",
            json.dumps({"id": "b", "input": "B", "expected": "neo4j", "split": "test", "tags": ["hard"]}),
        ]),
        encoding="utf-8",
    )
    examples = data.load_jsonl(path)
    assert [example.id for example in examples] == ["a", "b"]
    assert [example.id for example in data.select(examples, split="test", tag="hard")] == ["b"]
    assert len(data.sha256_file(path)) == 64


def test_new_row_contains_required_fields():
    row = data.new_row(id="x", tags=["hard"], custom=7)
    assert row["run_id"] is None
    assert row["id"] == "x"
    assert row["tags"] == ["hard"]
    assert row["custom"] == 7
