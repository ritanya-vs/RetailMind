import zlib

import numpy as np
import pytest

from retailmind.retrieval.dense import DenseRetriever, ids_fingerprint
from retailmind.retrieval.fusion import rrf
from retailmind.retrieval.tokenize import tokenize


class FakeEncoder:
    """Hashes words into a 64-dim bag-of-words vector, so texts sharing words are close."""

    def encode(self, texts, *, show_progress=False):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % 64] += 1.0
        return out


IDS = ["p1", "p2", "p3", "p4"]
TEXTS = [
    "red nike running shoes",
    "blue adidas running shoes",
    "red leather wallet",
    "steel bottle",
]


@pytest.fixture
def retriever():
    return DenseRetriever.build(IDS, TEXTS, FakeEncoder())


def test_most_similar_product_is_first_and_scores_descend(retriever):
    hits = retriever.search("red nike shoes", 3)
    assert hits[0][0] == "p1"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True) and len(hits) == 3


def test_scores_are_cosine_similarities(retriever):
    (pid, score), = retriever.search("steel bottle", 1)
    assert pid == "p4" and score == pytest.approx(1.0, abs=1e-5)


def test_batch_matches_single_search_and_blank_query_is_empty(retriever):
    batch = retriever.search_batch(["red wallet", "", "steel bottle"], 2)
    assert batch[0] == retriever.search("red wallet", 2)
    assert batch[1] == []
    assert batch[2][0][0] == "p4"


def test_k_larger_than_corpus_returns_everything(retriever):
    assert len(retriever.search("shoes", 50)) == 4


def test_save_and_load_round_trip(retriever, tmp_path):
    retriever.save(tmp_path / "idx", "fake-model")
    loaded = DenseRetriever.load(
        tmp_path / "idx", FakeEncoder(), expected_model="fake-model", expected_ids=IDS
    )
    assert loaded.search("red nike shoes", 3) == retriever.search("red nike shoes", 3)


def test_load_rejects_wrong_model_wrong_corpus_and_missing_index(retriever, tmp_path):
    retriever.save(tmp_path / "idx", "fake-model")
    with pytest.raises(ValueError, match="fake-model"):
        DenseRetriever.load(tmp_path / "idx", FakeEncoder(), expected_model="other")
    with pytest.raises(ValueError, match="different product set"):
        DenseRetriever.load(tmp_path / "idx", FakeEncoder(), expected_ids=IDS[:3])
    with pytest.raises(FileNotFoundError):
        DenseRetriever.load(tmp_path / "nothing", FakeEncoder())


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        DenseRetriever(["a"], np.zeros((2, 4), dtype=np.float32), FakeEncoder())
    with pytest.raises(ValueError):
        DenseRetriever([], np.zeros((0, 4), dtype=np.float32), FakeEncoder())


def test_fingerprint_depends_on_ids_and_order():
    assert ids_fingerprint(["a", "b"]) != ids_fingerprint(["b", "a"])
    assert ids_fingerprint(["a", "b"]) == ids_fingerprint(["a", "b"])


def test_rrf_scores_match_hand_calculation():
    first = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    second = [("b", 0.9), ("d", 0.8), ("a", 0.7)]
    fused = dict(rrf([first, second], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_orders_by_fused_score_and_respects_top_k():
    first = [("a", 1.0), ("b", 1.0)]
    second = [("b", 1.0), ("a", 1.0)]
    fused = rrf([first, second], k=60)
    assert [p for p, _ in fused] == ["a", "b"]  # exact tie: first seen wins
    assert rrf([first, second], k=60, top_k=1) == fused[:1]
    # a product in both lists beats products that appear in only one
    assert [p for p, _ in rrf([[("x", 1), ("y", 1)], [("z", 1), ("y", 1)]])][0] == "y"


def test_rrf_ignores_score_scale_and_handles_empty_input():
    small = rrf([[("a", 0.001), ("b", 0.0005)]])
    large = rrf([[("a", 1000.0), ("b", 500.0)]])
    assert small == large
    assert rrf([]) == [] and rrf([[], []]) == []
    with pytest.raises(ValueError):
        rrf([[("a", 1.0)]], k=0)
