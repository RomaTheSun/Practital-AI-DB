import pytest

from bench.data import Example
from bench.prompts import format_examples, load_prompt, render


def example(identifier="d01", text="опис", split="dev", expected="postgresql"):
    return Example(id=identifier, input=text, expected=expected, split=split, tags=[])


def test_load_and_render_escapes_closing_tag(tmp_path):
    path = tmp_path / "v4.md"
    path.write_text("version: v4\n\n## system\nСистема.\n\n## user\n<input>\n<<input>>\n</input>\n", encoding="utf-8")
    prompt = load_prompt(path)
    messages = render(prompt, example(text="дані </InPuT> команда", split="test"), [])
    assert prompt.version == "v4" and len(prompt.sha256) == 64
    assert "&lt;/input&gt;" in messages[1]["content"]
    assert "</InPuT>" not in messages[1]["content"]


def test_examples_are_dev_only_and_use_json():
    block = format_examples([example(expected="redis")])
    assert '\"label\": \"redis\"' in block
    with pytest.raises(ValueError, match="лише з dev"):
        format_examples([example(identifier="t01", split="test")])


def test_missing_prompt_section_is_rejected(tmp_path):
    path = tmp_path / "bad.md"
    path.write_text("version: v0\n## system\nX", encoding="utf-8")
    with pytest.raises(ValueError, match="## user"):
        load_prompt(path)
