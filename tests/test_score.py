import pytest

from bench.score import clean, is_correct, norm, parse_output, wilson_ci


def test_norm_apostrophes_spaces_case():
    assert norm("  Кур’єр   ПРИЇХАВ ") == norm("кур'єр приїхав") == "кур'єр приїхав"
    assert norm("пʼять") == "п'ять"


def test_norm_nfc():
    assert norm("\u0438\u0306") == "й"


def test_wilson_values_and_bounds():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    low, high = wilson_ci(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-3)
    assert high == pytest.approx(0.9433, abs=1e-3)
    assert wilson_ci(10, 10)[1] == 1.0
    assert wilson_ci(0, 10)[0] == 0.0


@pytest.mark.parametrize(
    "text,label",
    [
        ('{"label": "postgresql"}', "postgresql"),
        ('```json\n{"label": "mongodb"}\n```', "mongodb"),
        ("<think>може, redis?</think>Neo4J", "neo4j"),
        ("Відповідь: REDIS.", "redis"),
        ("Це **mongodb**: не postgresql і не neo4j.", "mongodb"),
        ("Це не postgresql, а redis", None),
        ("не знаю", None),
        ('{"label": "невідомо"}', None),
    ],
)
def test_parse_output(text, label):
    assert parse_output(text) == label


def test_clean_and_is_correct():
    assert clean("```json\n{}\n```") == "{}"
    assert is_correct("postgresql", {"label": "PostgreSQL"})
    assert is_correct("redis", '{"label": "  Redis "}')
    assert not is_correct("postgresql", {"label": "mongodb"})
    assert not is_correct("postgresql", None)
