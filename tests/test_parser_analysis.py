import pandas as pd
import pytest

from retailmind.query.analysis import agreement, group_judgments, product_info, satisfies
from retailmind.query.lexicon import Lexicon
from retailmind.query.parser import QueryParser

PRODUCTS = pd.DataFrame(
    {
        "product_id": ["p1", "p2", "p3", "p4", "p5"],
        "brand": ["Nike", "Nike", "Adidas", "", ""],
        "color": ["Black", "Red", "Black/White", "", ""],
        "text": [
            "Nike black running shoes",
            "Nike red shoes with wheels",
            "Adidas black white sneakers",
            "plain shoes",
            "black Nike shoes",
        ],
    }
)
QRELS = pd.DataFrame(
    {
        "query_id": ["q1"] * 4 + ["q2"] * 2,
        "product_id": ["p1", "p2", "p3", "p4", "p1", "p2"],
        "label": ["E", "S", "I", "I", "E", "E"],
    }
)
LEXICON = Lexicon(brands=frozenset({"nike", "adidas"}), colors=frozenset({"black", "red"}))


@pytest.fixture
def parsed():
    parser = QueryParser(LEXICON)
    return {
        "q1": parser.parse("nike black shoes", "q1"),
        "q2": parser.parse("shoes without wheels", "q2"),
    }


def test_brand_agreement_by_label(parsed):
    result = agreement(parsed, group_judgments(QRELS), product_info(PRODUCTS), "brand")
    assert result["queries"] == ["q1"]
    # q1 judged: p1 E (Nike), p2 S (Nike), p3 I (Adidas), p4 I (no brand)
    assert result["counts"]["E"] == [1, 1] and result["counts"]["S"] == [1, 1]
    assert result["counts"]["I"] == [0, 2]
    assert result["rates"]["I"] == 0.0 and result["rates"]["C"] is None


def test_color_matches_compound_colors(parsed):
    info = product_info(PRODUCTS)
    assert satisfies("color", ["black"], info["p3"])  # "Black/White" contains black
    assert not satisfies("color", ["black"], info["p2"])


def test_excluded_word_agreement(parsed):
    result = agreement(parsed, group_judgments(QRELS), product_info(PRODUCTS), "exclude_terms")
    assert result["queries"] == ["q2"]
    assert result["counts"]["E"] == [1, 2]  # only p2 mentions wheels


def test_group_judgments_groups_by_query():
    grouped = group_judgments(QRELS)
    assert grouped["q2"] == [("p1", "E"), ("p2", "E")]


def test_title_text_counts_only_when_asked(parsed):
    info = product_info(PRODUCTS)
    assert not satisfies("color", ["black"], info["p5"])
    assert satisfies("color", ["black"], info["p5"], use_text=True)
    assert not satisfies("brand", ["nike"], info["p5"])
    assert satisfies("brand", ["nike"], info["p5"], use_text=True)
