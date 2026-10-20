from bench.baseline import predict


def test_baseline_returns_majority_label():
    assert predict("будь-який опис") == "postgresql"
