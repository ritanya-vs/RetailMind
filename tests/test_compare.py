import pytest

from retailmind.eval.compare import paired_comparison


def test_clear_improvement_has_an_interval_above_zero():
    a = {str(i): 0.4 for i in range(50)}
    b = {str(i): 0.5 + (0.01 if i % 2 else 0.0) for i in range(50)}
    result = paired_comparison(a, b)
    assert result["mean_diff"] == pytest.approx(0.105)
    assert result["ci_low"] > 0 and result["wins"] == 50 and result["losses"] == 0


def test_no_difference_has_an_interval_around_zero():
    a = {str(i): float(i % 2) for i in range(60)}
    b = {str(i): float((i + 1) % 2) for i in range(60)}  # swaps wins and losses evenly
    result = paired_comparison(a, b)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["wins"] == result["losses"] == 30


def test_is_reproducible_and_rejects_mismatched_queries():
    a, b = {"1": 0.1, "2": 0.9}, {"1": 0.3, "2": 0.5}
    assert paired_comparison(a, b) == paired_comparison(a, b)
    with pytest.raises(ValueError):
        paired_comparison({"1": 0.1}, {"2": 0.1})
    with pytest.raises(ValueError):
        paired_comparison({}, {})
