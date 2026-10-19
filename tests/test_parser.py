import pandas as pd
import pytest

from retailmind.query.lexicon import Lexicon, LexiconBuilder, tokens
from retailmind.query.parser import QueryParser

LEXICON = Lexicon(
    brands=frozenset({"nike", "adidas", "amazon basics", "under armour", "sony"}),
    colors=frozenset({"black", "red", "navy", "rose gold", "blue"}),
)


@pytest.fixture
def parser():
    return QueryParser(LEXICON)


def test_documented_example_is_parsed_fully(parser):
    parsed = parser.parse("black nike running shoes under $60 delivered in 3 days", "q1")
    c = parsed.constraints
    assert c.brand == ["nike"] and c.color == ["black"]
    assert c.price_max == 60.0 and c.price_min is None
    assert c.delivery_max_days == 3
    assert parsed.product_terms == "black nike running shoes"
    assert parsed.query_id == "q1" and parsed.raw.startswith("black nike")


def test_multiword_brand_and_color_use_the_longest_match(parser):
    c = parser.parse("amazon basics rose gold phone cable").constraints
    assert c.brand == ["amazon basics"] and c.color == ["rose gold"]
    assert parser.parse("Under Armour hoodie").constraints.brand == ["under armour"]


def test_unknown_words_are_not_constraints(parser):
    assert parser.parse("organic green tea bags").constraints.is_empty()


@pytest.mark.parametrize(
    ("query", "low", "high"),
    [
        ("shoes under $50", None, 50.0),
        ("shoes below 50 dollars", None, 50.0),
        ("shoes less than $49.99", None, 49.99),
        ("shoes up to $1,200", None, 1200.0),
        ("shoes over $20", 20.0, None),
        ("shoes at least 20 usd", 20.0, None),
        ("shoes between $20 and $50", 20.0, 50.0),
        ("shoes $20-$50", 20.0, 50.0),
        ("shoes $50 to $20", 20.0, 50.0),
    ],
)
def test_price_phrases(parser, query, low, high):
    c = parser.parse(query).constraints
    assert (c.price_min, c.price_max) == (low, high)


def test_numbers_without_currency_are_not_prices(parser):
    c = parser.parse("backpack under 50 lbs").constraints
    assert c.price_max is None and c.size == ["50 lbs"]


@pytest.mark.parametrize(
    ("query", "days"),
    [
        ("headphones delivered in 2 days", 2),
        ("headphones arrive within 3 business days", 3),
        ("headphones 2 day delivery", 2),
        ("headphones next day delivery", 1),
        ("headphones delivered tomorrow", 1),
        ("headphones same day shipping", 0),
        ("headphones delivered within a week", 7),
    ],
)
def test_delivery_phrases(parser, query, days):
    assert parser.parse(query).constraints.delivery_max_days == days
    assert "deliver" not in parser.parse(query).product_terms


def test_stock_and_soft_preferences(parser):
    parsed = parser.parse("cheap headphones in stock")
    assert parsed.constraints.in_stock_only is True
    assert parsed.soft_preferences == ["low price"]
    assert "stock" not in parsed.product_terms


@pytest.mark.parametrize(
    ("query", "sizes"),
    [
        ("shampoo 16 oz", ["16 oz"]),
        ("pillow cover 18x18", ["18x18"]),
        ("box 10 x 10 x 8", ["10x10x8"]),
        ("t shirt size 10", ["size 10"]),
        ("hoodie xl", ["xl"]),
        ("hoodie 2xl black", ["2xl"]),
        ("water bottles 24 pack", ["24 pack"]),
        ("2 in 1 shampoo", []),
    ],
)
def test_sizes(parser, query, sizes):
    assert parser.parse(query).constraints.size == sizes


def test_materials(parser):
    assert parser.parse("stainless steel water bottle").constraints.material == ["stainless steel"]
    assert parser.parse("leather wallet").constraints.material == ["leather"]


def test_exclusions_are_removed_from_the_product_terms(parser):
    parsed = parser.parse("white desk chair without wheels")
    assert parsed.constraints.exclude_terms == ["wheels"]
    assert parsed.product_terms == "white desk chair"
    parsed = parser.parse("water heater not adapter")
    assert parsed.constraints.exclude_terms == ["adapter"]
    assert parser.parse("pillow with no smell").constraints.exclude_terms == ["smell"]
    assert parser.parse("tea without any sugar").constraints.exclude_terms == ["sugar"]


def test_exclusion_edge_cases(parser):
    assert parser.parse("shoes not").constraints.exclude_terms == []
    assert parser.parse("room no 5").constraints.exclude_terms == []
    # a named exclusion must not also count as a wanted brand or color
    c = parser.parse("sneakers not black").constraints
    assert c.exclude_terms == ["black"] and c.color == []


def test_empty_and_symbol_queries(parser):
    assert parser.parse("").constraints.is_empty()
    assert parser.parse("!!! ,,,").product_terms == ""


def test_tokens_fold_accents_and_case():
    assert tokens("L'Oréal Paris") == ["l", "oreal", "paris"]
    assert tokens(None) == [] and tokens(float("nan")) == []


def test_lexicon_builder_keeps_clear_brands_and_drops_ambiguous_words():
    builder = LexiconBuilder()
    brands = ["Nike"] * 10 + ["Apple"] * 6 + ["Amazon Basics"] * 7 + ["Rare"] * 2 + ["Women"] * 8
    colors = (
        ["Black"] * 60 + ["Navy Blue"] * 55 + ["Teal"] * 10 + ["Black/White"] * 52 + ["XL"] * 99
    )
    titles = (
        ["nike shoes"] * 10  # nike appears only for the brand
        + ["case for apple iphone"] * 40  # apple appears mostly for other brands' accessories
        + ["amazon basics cable"] * 7
        + ["women dress"] * 100
    )
    builder.update(brands, colors, titles)
    lexicon = builder.build(min_brand_count=5, min_precision=0.3, min_color_count=50)
    assert "nike" in lexicon.brands and "amazon basics" in lexicon.brands
    assert "apple" not in lexicon.brands  # 6 / 40 titles = 0.15 < 0.3
    assert "women" not in lexicon.brands  # generic word
    assert "rare" not in lexicon.brands  # too few products
    assert lexicon.colors == {
        "black",
        "navy blue",
        "black white",
    }  # teal too rare, xl too short, glass and steel are not colors


def test_lexicon_save_and_load_round_trip(tmp_path):
    LEXICON.save(tmp_path / "lex.json")
    loaded = Lexicon.load(tmp_path / "lex.json")
    assert loaded.brands == LEXICON.brands and loaded.colors == LEXICON.colors
    assert "leather" in loaded.materials


def test_products_frame_can_feed_the_builder():
    frame = pd.DataFrame({"b": ["Sony"] * 50, "c": ["Black"] * 50, "t": ["sony tv"] * 50})
    builder = LexiconBuilder()
    builder.update(frame["b"], frame["c"], frame["t"])
    assert builder.n_products == 50 and "sony" in builder.build().brands
