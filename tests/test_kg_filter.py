import pandas as pd
import pytest

from retailmind.data.synthetic import synthetic_attributes
from retailmind.kg.filter import ConstraintFilter
from retailmind.kg.graph import ProductKG
from retailmind.query.lexicon import Lexicon
from retailmind.schemas import Constraints

LEXICON = Lexicon(
    brands=frozenset({"nike", "adidas", "amazon basics"}),
    colors=frozenset({"black", "red", "blue", "black white"}),
)
PRODUCTS = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4"],
        "brand": ["Nike", "Adidas", "", "Nike"],
        "color": ["Black", "", "Red", "Blue"],
        "text": [
            "Nike running shoes black leather",
            "Adidas shoes in red",
            "Amazon Basics blue cotton shirt",
            "Nike hoodie",
        ],
    }
)
ATTRIBUTES = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4"],
        "price": [30.0, 80.0, 10.0, 55.0],
        "delivery_days": [2, 5, 1, 3],
        "in_stock": [True, True, False, True],
    }
)


@pytest.fixture(scope="module")
def kg():
    return ProductKG.from_products(PRODUCTS, ATTRIBUTES, LEXICON)


def test_graph_links_products_to_entities_from_fields_and_text(kg):
    assert kg.n_products == 4
    assert kg.entities("p1", "brand") == {"nike"}
    assert kg.entities("p1", "color") == {"black"}
    assert kg.entities("p1", "material") == {"leather"}
    assert kg.entities("p2", "color") == {"red"}  # the color field is empty, found in the text
    assert kg.entities("p3", "brand") == {"amazon basics"}  # the brand field is empty
    assert kg.products_with("brand", "nike") == {"p1", "p4"}
    assert kg.products_with("color", "purple") == set()


def test_each_constraint_kind_is_checked(kg):
    everything = {"price", "delivery", "in_stock", "brand", "color", "material"}
    c = Constraints(
        brand=["nike"], color=["black"], material=["leather"], price_max=40,
        delivery_max_days=3, in_stock_only=True,
    )  # fmt: skip
    assert kg.violations("p1", c, everything) == []
    assert kg.violations("p2", c, everything) == ["price", "delivery", "brand", "color", "material"]
    assert kg.violations("p3", c, everything) == ["in_stock", "brand", "color", "material"]
    assert kg.violations("p4", c, everything) == ["price", "color", "material"]


def test_price_minimum_and_disabled_kinds(kg):
    assert kg.violations("p3", Constraints(price_min=20), {"price"}) == ["price"]
    assert kg.violations("p3", Constraints(price_min=20), {"delivery"}) == []
    assert kg.violations("p2", Constraints(brand=["nike"]), set()) == []


def test_strict_mode_removes_violators():
    flt = ConstraintFilter(_kg(), mode="strict", enabled=["price", "delivery"])
    ranked = [("p2", 0.9), ("p1", 0.8), ("p4", 0.7)]
    kept, violations = flt.apply(Constraints(price_max=60, delivery_max_days=3), ranked)
    assert [p for p, _ in kept] == ["p1", "p4"]
    assert violations == {"p2": ["price", "delivery"]}


def test_lenient_mode_demotes_by_the_penalty_per_violation():
    flt = ConstraintFilter(_kg(), mode="lenient", penalty=0.5, enabled=["price", "delivery"])
    ranked = [("p2", 0.9), ("p1", 0.8), ("p4", 0.7)]
    out, _ = flt.apply(Constraints(price_max=60, delivery_max_days=3), ranked)
    # p2 loses 2 x 0.5 and falls to 0.9 - 1.0 = -0.1, below both compliant products
    assert [p for p, _ in out] == ["p1", "p4", "p2"]
    assert dict(out)["p2"] == pytest.approx(-0.1)
    # a small penalty only reorders close scores
    gentle = ConstraintFilter(_kg(), mode="lenient", penalty=0.05, enabled=["price"])
    out, _ = gentle.apply(Constraints(price_max=60), [("p2", 0.9), ("p1", 0.8)])
    assert [p for p, _ in out] == ["p2", "p1"]


def test_hard_penalty_sinks_price_violators_while_soft_penalty_only_nudges():
    flt = ConstraintFilter(_kg(), penalty=0.5, hard_penalty=100.0, enabled=["price", "brand"])
    # p2: Adidas at $80; p1: Nike at $30; p4: Nike at $55
    ranked = [("p2", 9.0), ("p4", 8.0), ("p1", 7.0)]
    out, violations = flt.apply(Constraints(price_max=60, brand=["nike"]), ranked)
    assert violations == {"p2": ["price", "brand"]}
    assert [p for p, _ in out] == ["p4", "p1", "p2"]
    assert dict(out)["p2"] == pytest.approx(9.0 - 100.5)
    # a brand miss alone costs only the soft penalty
    out, _ = flt.apply(Constraints(brand=["adidas"]), [("p1", 9.2), ("p2", 9.0)])
    assert [p for p, _ in out] == ["p2", "p1"] and dict(out)["p1"] == pytest.approx(8.7)


def test_negative_penalties_are_rejected():
    with pytest.raises(ValueError):
        ConstraintFilter(_kg(), hard_penalty=-1)


def test_no_constraints_or_nothing_enabled_leaves_the_ranking_alone():
    ranked = [("p2", 0.9), ("p1", 0.8)]
    assert ConstraintFilter(_kg(), enabled=["price"]).apply(Constraints(), ranked)[0] == ranked
    assert ConstraintFilter(_kg(), enabled=[]).apply(Constraints(price_max=1), ranked)[0] == ranked


def test_unknown_products_pass_through_and_bad_arguments_fail():
    flt = ConstraintFilter(_kg(), mode="strict", enabled=["price"])
    kept, _ = flt.apply(Constraints(price_max=1), [("unknown", 1.0)])
    assert kept == [("unknown", 1.0)]
    with pytest.raises(ValueError):
        ConstraintFilter(_kg(), mode="medium")
    with pytest.raises(ValueError):
        ConstraintFilter(_kg(), penalty=-1)


def _kg():
    return ProductKG.from_products(PRODUCTS, ATTRIBUTES, LEXICON)


def test_synthetic_attributes_are_stable_and_in_range():
    ids = [f"B{i:05d}" for i in range(2000)]
    first = synthetic_attributes(ids, seed=42)
    again = synthetic_attributes(list(reversed(ids)), seed=42).set_index("product_id").loc[ids]
    pd.testing.assert_frame_equal(first.set_index("product_id"), again)  # order does not matter
    assert first["price"].between(1, 800).all()
    assert set(first["delivery_days"]) <= {1, 2, 3, 4, 5, 7, 10}
    assert 0.85 < first["in_stock"].mean() < 0.95
    assert 20 < first["price"].median() < 30
    other = synthetic_attributes(ids, seed=7)
    assert not first["price"].equals(other["price"])
