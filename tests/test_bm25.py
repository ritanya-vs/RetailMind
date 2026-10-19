import pytest

from retailmind.retrieval.bm25 import BM25Retriever
from retailmind.retrieval.tokenize import tokenize


def test_tokenize_lowercases_and_splits_on_symbols():
    assert tokenize("Nike Air-Max 90, Men's!") == ["nike", "air", "max", "90", "men", "s"]
    assert tokenize("") == []


@pytest.fixture
def index():
    ids = ["p1", "p2", "p3", "p4"]
    texts = [
        "red nike running shoes",
        "blue adidas running shoes",
        "red leather wallet",
        "stainless steel water bottle",
    ]
    return BM25Retriever(ids, texts)


def test_best_match_is_ranked_first(index):
    hits = index.search("red nike shoes", 3)
    assert hits[0][0] == "p1"
    assert [score for _, score in hits] == sorted((s for _, s in hits), reverse=True)


def test_products_without_any_matching_term_are_left_out(index):
    ids = [pid for pid, _ in index.search("wallet", 10)]
    assert ids == ["p3"]


def test_k_limits_results(index):
    assert len(index.search("red running shoes", 2)) == 2


def test_empty_or_unknown_query_returns_nothing(index):
    assert index.search("", 5) == []
    assert index.search("!!!", 5) == []
    assert index.search("zzzzz", 5) == []


def test_ties_are_broken_by_corpus_order():
    retriever = BM25Retriever(["a", "b", "c"], ["shoe", "shoe", "shoe boot"])
    first = [pid for pid, _ in retriever.search("shoe", 3)]
    assert first[:2] == ["a", "b"] and first == [pid for pid, _ in retriever.search("shoe", 3)]


def test_mismatched_or_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever(["a"], ["x", "y"])
    with pytest.raises(ValueError):
        BM25Retriever([], [])
