import numpy as np
import pandas as pd

from retailmind.config import load_config
from retailmind.generate.llm import ScriptedClient
from retailmind.kg.graph import ProductKG
from retailmind.pipeline import Pipeline
from retailmind.query.lexicon import Lexicon

PRODUCTS = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4", "p5"],
        "title": ["", "", "", "", ""],
        "brand": ["Nike", "Adidas", "Nike", "", ""],
        "color": ["Black", "Blue", "Red", "", ""],
        "text": [
            "nike black running shoes",
            "adidas blue running shoes",
            "nike red basketball shoes",
            "stainless steel water bottle",
            "yoga mat purple",
        ],
    }
)
ATTRIBUTES = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4", "p5"],
        "price": [30.0, 90.0, 50.0, 15.0, 20.0],
        "delivery_days": [2, 5, 3, 1, 2],
        "in_stock": [True, True, False, True, True],
    }
)
LEXICON = Lexicon(brands=frozenset({"nike", "adidas"}), colors=frozenset({"black", "blue", "red"}))


class FakeEncoder:
    def encode(self, texts, *, show_progress=False):
        out = np.zeros((len(texts), 16), dtype=np.float32)
        vocab = "nike adidas black blue red running shoes bottle yoga mat steel water".split()
        for row, text in enumerate(texts):
            for i, word in enumerate(vocab):
                if word in text.lower():
                    out[row, i % 16] += 1.0
        return out


def _pipeline(components, llm=None, monkeypatch=None):
    cfg = load_config()
    kg = ProductKG.from_products(PRODUCTS, ATTRIBUTES, LEXICON) if "filter" in components else None
    return Pipeline(cfg, set(components), PRODUCTS, lexicon=LEXICON, kg=kg, llm=llm)


def test_bm25_only_pipeline_retrieves_and_has_no_answer():
    pipe = _pipeline({"bm25"})
    response = pipe.search("nike running shoes", "q1")
    assert response.hits and response.hits[0].rank == 1
    assert response.answer is None and response.parsed is None


def test_parse_and_filter_remove_or_demote_violators():
    pipe = _pipeline({"bm25", "parse", "filter"})
    response = pipe.search("nike running shoes under $40")
    assert response.parsed.constraints.price_max == 40.0
    # p2 (adidas, $90) and p3 ($50) break price; with the hard penalty they sink below p1 ($30)
    assert response.hits[0].product_id == "p1"


def test_generate_grounds_the_answer_on_retrieved_products():
    llm = ScriptedClient(["Try [1], it is a solid pick."])
    pipe = _pipeline({"bm25", "parse", "generate"}, llm=llm)
    response = pipe.search("nike black shoes")
    assert response.answer == "Try [1], it is a solid pick."
    assert "nike black running shoes" in llm.calls[0][1]["content"]


def test_decompose_retrieves_for_each_sub_need():
    llm = ScriptedClient(['["water bottle", "yoga mat"]', "Here are [1] and [2]."])
    pipe = _pipeline({"bm25", "decompose", "generate"}, llm=llm)
    response = pipe.search("a water bottle and a yoga mat")
    found = {hit.product_id for hit in response.hits}
    assert "p4" in found and "p5" in found  # both needs are represented


def test_full_agent_pipeline_runs_and_traces():
    llm = ScriptedClient(default="A grounded recommendation [1].")
    # dense and rerank need heavy models/indexes (tested on their own); this checks the control flow
    pipe = _pipeline({"bm25", "parse", "filter", "decompose", "agent", "generate"}, llm=llm)
    response = pipe.search("nike shoes under $60")
    assert response.answer and response.iterations >= 1
    assert response.trace and response.trace[-1] == "generate"


def test_pipeline_from_config_skips_llm_when_not_needed(monkeypatch, tmp_path):
    # from_config must not try to build an LLM client for an LLM-free ablation
    cfg = load_config()
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: PRODUCTS)
    pipe = Pipeline.from_config(cfg, {"bm25"})
    assert pipe.generator is None and pipe.decomposer is None
