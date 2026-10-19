import numpy as np
import pytest

from retailmind.rerank.cross_encoder import Reranker
from retailmind.retrieval.tokenize import tokenize

TEXTS = {
    "a": "steel water bottle",
    "b": "red nike running shoes",
    "c": "nike running shoes blue",
    "d": "leather wallet",
    "e": "red shoes",
}


class OverlapScorer:
    """Scores a pair by how many query words appear in the document."""

    def __init__(self):
        self.calls = 0
        self.pairs_seen = 0

    def score(self, pairs, *, show_progress=False):
        self.calls += 1
        self.pairs_seen += len(pairs)
        return np.array([len(set(tokenize(q)) & set(tokenize(d))) for q, d in pairs], dtype=float)


@pytest.fixture
def reranker():
    return Reranker(OverlapScorer(), TEXTS)


def test_top_candidates_are_reordered_by_the_scorer(reranker):
    candidates = [("a", 9.0), ("c", 8.0), ("b", 7.0)]
    ranked = reranker.rerank("red nike running shoes", candidates, top_k=3)
    assert [pid for pid, _ in ranked] == ["b", "c", "a"]
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_candidates_beyond_top_k_stay_below_in_original_order(reranker):
    candidates = [("a", 5.0), ("b", 4.0), ("d", 3.0), ("e", 2.0), ("c", 1.0)]
    ranked = reranker.rerank("red nike running shoes", candidates, top_k=2)
    assert [pid for pid, _ in ranked][:2] == ["b", "a"]
    assert [pid for pid, _ in ranked][2:] == ["d", "e", "c"]  # untouched tail
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores)


def test_only_top_k_pairs_are_scored(reranker):
    reranker.rerank("shoes", [(p, 1.0) for p in TEXTS], top_k=2)
    assert reranker.scorer.pairs_seen == 2


def test_batch_scores_everything_in_one_call_and_matches_single(reranker):
    queries = ["red shoes", "wallet"]
    lists = [[("a", 2.0), ("e", 1.0)], [("a", 2.0), ("d", 1.0)]]
    batch = reranker.rerank_batch(queries, lists, top_k=5)
    assert reranker.scorer.calls == 1
    assert batch[0] == reranker.rerank(queries[0], lists[0], 5)
    assert batch[1][0][0] == "d"


def test_ties_keep_the_original_order(reranker):
    ranked = reranker.rerank("zzz", [("d", 3.0), ("a", 2.0), ("e", 1.0)], top_k=3)
    assert [pid for pid, _ in ranked] == ["d", "a", "e"]


def test_empty_inputs_and_bad_arguments(reranker):
    assert reranker.rerank("shoes", [], top_k=5) == []
    assert reranker.rerank_batch([], [], top_k=5) == []
    with pytest.raises(ValueError):
        reranker.rerank("shoes", [("a", 1.0)], top_k=0)
    with pytest.raises(KeyError, match="nope"):
        reranker.rerank("shoes", [("nope", 1.0)], top_k=1)


def test_wrong_number_of_scores_is_rejected():
    class Broken:
        def score(self, pairs, *, show_progress=False):
            return np.zeros(len(pairs) + 1)

    with pytest.raises(ValueError, match="different number"):
        Reranker(Broken(), TEXTS).rerank("shoes", [("a", 1.0)], top_k=1)
