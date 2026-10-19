import pandas as pd
import pytest

from retailmind.eval.metrics import evaluate_results, load_split


def _qrels(rows):
    frame = pd.DataFrame(rows, columns=["query_id", "product_id", "label"])
    frame["gain"] = frame["label"].map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})
    return frame


QRELS = _qrels(
    [("q2", "a", "E"), ("q2", "b", "S"), ("q2", "c", "I"), ("q1", "x", "E"), ("q1", "y", "S")]
)


def test_perfect_ranking_scores_one():
    results = {"q1": [("x", 3.0), ("y", 2.0)], "q2": [("a", 3.0), ("b", 2.0)]}
    means, _ = evaluate_results(results, QRELS)
    assert means["ndcg@10"] == pytest.approx(1.0)
    assert means["mrr@10"] == pytest.approx(1.0)
    assert means["recall@100"] == pytest.approx(1.0)


def test_graded_ndcg_matches_hand_calculation_per_query():
    # q2 ranks the Substitute above the Exact match:
    # dcg = 0.1/log2(2) + 1/log2(3); ideal = 1 + 0.1/log2(3)
    results = {"q1": [("x", 3.0), ("y", 2.0)], "q2": [("b", 3.0), ("a", 2.0)]}
    means, by_query = evaluate_results(results, QRELS)
    assert by_query["ndcg@10"]["q1"] == pytest.approx(1.0)
    assert by_query["ndcg@10"]["q2"] == pytest.approx(0.6876, abs=1e-3)
    assert by_query["mrr@10"]["q2"] == pytest.approx(0.5)
    assert means["ndcg@10"] == pytest.approx((1.0 + 0.6876) / 2, abs=1e-3)


def test_only_exact_counts_for_binary_metrics():
    results = {"q1": [("y", 2.0)], "q2": [("b", 2.0)]}  # only substitutes retrieved
    means, _ = evaluate_results(results, QRELS)
    assert means["recall@100"] == 0.0 and means["mrr@10"] == 0.0
    assert means["ndcg@10"] > 0.0  # graded nDCG still gives partial credit


def test_missing_and_empty_results_score_zero():
    means, by_query = evaluate_results({"q1": [("x", 1.0)], "q2": []}, QRELS)
    assert by_query["ndcg@10"]["q2"] == 0.0
    # q1 finds only the Exact item: 1 / (1 + 0.1/log2(3)) = 0.9406, q2 scores 0
    assert means["ndcg@10"] == pytest.approx(0.9406 / 2, abs=1e-3)
    means, _ = evaluate_results({}, QRELS)
    assert means["ndcg@10"] == 0.0


def test_load_split_returns_only_that_splits_queries(tmp_path):
    pd.DataFrame(
        {"query_id": [1, 2], "query": ["a", "b"], "split": ["dev", "test"],
         "n_judged": [1, 1], "n_exact": [1, 1]}
    ).to_parquet(tmp_path / "queries.parquet")
    (tmp_path / "qrels.tsv").write_text(
        "query_id\tproduct_id\tlabel\tgain\n1\tp1\tE\t1.0\n2\tp2\tE\t1.0\n", encoding="utf-8"
    )
    queries, qrels = load_split(tmp_path, "dev")
    assert queries["query_id"].tolist() == ["1"]
    assert qrels["product_id"].tolist() == ["p1"]
    with pytest.raises(ValueError):
        load_split(tmp_path, "nope")
