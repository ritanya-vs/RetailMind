import pandas as pd
import pytest

from retailmind.data.synthetic import synthetic_attributes
from retailmind.eval.synthetic import (
    TEMPLATES,
    load_sc,
    make_sc_set,
    violation_rates,
)
from retailmind.kg.graph import ProductKG
from retailmind.query.lexicon import Lexicon
from retailmind.query.parser import QueryParser
from retailmind.schemas import Constraints


def _dataset(n_queries: int = 60):
    queries, rows, product_ids = [], [], []
    for q in range(n_queries):
        queries.append({"query_id": str(q), "query": f"base query {q}"})
        for j, label in enumerate(["E"] * 6 + ["S"] * 3 + ["I"] * 2):
            product_id = f"B{q:03d}{j:02d}"
            product_ids.append(product_id)
            rows.append({"query_id": str(q), "product_id": product_id, "label": label})
    return pd.DataFrame(queries), pd.DataFrame(rows), synthetic_attributes(product_ids, seed=1)


@pytest.fixture(scope="module")
def built():
    queries, qrels, attributes = _dataset()
    sc_queries, sc_qrels = make_sc_set(queries, qrels, attributes, n=30, seed=3)
    return queries, qrels, attributes, sc_queries, sc_qrels


def test_builds_the_requested_number_with_all_templates(built):
    _, _, _, sc_queries, _ = built
    assert len(sc_queries) == 30 and sc_queries["query_id"].is_unique
    assert set(sc_queries["template"]) == {"+".join(t) for t in TEMPLATES}


def test_every_query_has_exact_products_that_pass_and_fail(built):
    _, qrels, attributes, sc_queries, sc_qrels = built
    for query_id, base_id in zip(sc_queries["query_id"], sc_queries["base_query_id"], strict=True):
        original = qrels[(qrels["query_id"] == base_id) & (qrels["label"] == "E")]
        kept = sc_qrels[(sc_qrels["query_id"] == query_id) & (sc_qrels["label"] == "E")]
        assert 0 < len(kept) < len(original)


def test_relevance_gains_follow_the_constraints(built):
    _, qrels, attributes, sc_queries, sc_qrels = built
    row = sc_queries.iloc[0]
    truth = Constraints.model_validate_json(row["constraints"])
    attrs = attributes.set_index("product_id")
    for rec in sc_qrels[sc_qrels["query_id"] == row["query_id"]].itertuples():
        info = attrs.loc[rec.product_id]
        meets = (
            (truth.price_max is None or info["price"] <= truth.price_max)
            and (
                truth.delivery_max_days is None or info["delivery_days"] <= truth.delivery_max_days
            )
            and (not truth.in_stock_only or info["in_stock"])
        )
        if not meets:
            assert rec.label == "I" and rec.gain == 0.0


def test_the_parser_recovers_every_generated_constraint(built):
    _, _, _, sc_queries, _ = built
    parser = QueryParser(Lexicon())
    for text, raw in zip(sc_queries["query"], sc_queries["constraints"], strict=True):
        truth = Constraints.model_validate_json(raw)
        parsed = parser.parse(text).constraints
        assert parsed.price_max == truth.price_max, text
        assert parsed.delivery_max_days == truth.delivery_max_days, text
        assert parsed.in_stock_only == truth.in_stock_only, text
        assert parser.parse(text).product_terms.startswith("base query"), text
        assert "$" not in parser.parse(text).product_terms


def test_is_deterministic_and_reports_when_it_cannot_build(built):
    queries, qrels, attributes, sc_queries, sc_qrels = built
    again_q, again_r = make_sc_set(queries, qrels, attributes, n=30, seed=3)
    pd.testing.assert_frame_equal(sc_queries, again_q)
    pd.testing.assert_frame_equal(sc_qrels, again_r)
    with pytest.raises(ValueError, match="could only build"):
        make_sc_set(queries, qrels, attributes, n=1000, seed=3)


def test_load_sc_round_trip(built, tmp_path):
    _, _, _, sc_queries, sc_qrels = built
    sc_queries.to_parquet(tmp_path / "sc_dev_queries.parquet", index=False)
    sc_qrels.to_csv(tmp_path / "sc_dev_qrels.tsv", sep="\t", index=False)
    queries, qrels, truth = load_sc(tmp_path, "dev")
    assert len(queries) == 30 and set(truth) == set(queries["query_id"])
    assert isinstance(next(iter(truth.values())), Constraints)
    assert set(qrels["query_id"]) == set(queries["query_id"])


def test_violation_rates_count_broken_constraints_in_the_top_k():
    products = pd.DataFrame(
        {"product_id": ["a", "b", "c"], "brand": ["", "", ""], "color": ["", "", ""],
         "text": ["x", "y", "z"]}
    )  # fmt: skip
    attributes = pd.DataFrame(
        {"product_id": ["a", "b", "c"], "price": [10.0, 90.0, 20.0],
         "delivery_days": [1, 1, 1], "in_stock": [True, True, True]}
    )  # fmt: skip
    kg = ProductKG.from_products(products, attributes, Lexicon())
    truth = {"q1": Constraints(price_max=50.0), "q2": Constraints(price_max=50.0)}
    results = {"q1": [("a", 1.0), ("c", 0.9)], "q2": [("b", 1.0), ("a", 0.9)]}
    rates = violation_rates(results, truth, kg, k=2)
    assert rates["violation@2"] == pytest.approx(0.25)  # 0 of 2 in q1, 1 of 2 in q2
    assert rates["clean@2"] == pytest.approx(0.5)
