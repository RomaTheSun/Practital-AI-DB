from bench.score import is_correct, norm, parse_output, wilson_ci


def test_norm_and_parse_output():
    assert norm("  Neo4J\n") == "neo4j"
    assert parse_output("neo4j") == "neo4j"
    assert parse_output("Відповідь: REDIS.") == "redis"
    assert parse_output("postgresql або redis") is None
    assert parse_output("без допустимої мітки") is None


def test_correctness_and_wilson_interval():
    assert is_correct("PostgreSQL", "postgresql")
    assert not is_correct("mongodb", None)
    low, high = wilson_ci(5, 10)
    assert 0 < low < 0.5 < high < 1
