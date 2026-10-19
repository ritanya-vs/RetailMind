import pytest

from retailmind.generate.answer import AnswerGenerator, build_context, build_messages
from retailmind.generate.llm import ScriptedClient
from retailmind.query.decompose import QueryDecomposer
from retailmind.schemas import Constraints

PRODUCTS = {
    "p1": {"title": "Nike Air Zoom", "text": "Nike Air Zoom black running shoes"},
    "p2": {"title": "Adidas Ultraboost", "text": "Adidas Ultraboost blue running shoes"},
    "p3": {"title": "Steel Bottle", "text": "stainless steel water bottle 24oz"},
}


def test_build_context_numbers_known_products_in_order():
    context, shown = build_context(["p2", "missing", "p1"], PRODUCTS, max_products=5)
    assert shown == ["p2", "p1"]
    assert context.splitlines()[0].startswith("[1] Adidas Ultraboost")
    assert context.splitlines()[1].startswith("[2] Nike Air Zoom")


def test_build_context_respects_max_products():
    _, shown = build_context(["p1", "p2", "p3"], PRODUCTS, max_products=2)
    assert shown == ["p1", "p2"]


def test_messages_include_query_and_constraint_line():
    msgs = build_messages("running shoes", "[1] x", Constraints(price_max=80, color=["black"]))
    user = msgs[1]["content"]
    assert "running shoes" in user and "max price $80" in user and "color: black" in user
    assert msgs[0]["role"] == "system"


def test_generator_grounds_on_shown_products_and_returns_them():
    client = ScriptedClient(["I recommend [1] for running."])
    gen = AnswerGenerator(client)
    answer, shown = gen.generate("running shoes", ["p1", "p2"], PRODUCTS, max_products=2)
    assert answer == "I recommend [1] for running." and shown == ["p1", "p2"]
    assert "[1] Nike Air Zoom" in client.calls[0][1]["content"]


def test_generator_handles_no_products_without_calling_the_llm():
    client = ScriptedClient(["should not be used"])
    answer, shown = AnswerGenerator(client).generate("x", ["unknown"], PRODUCTS)
    assert shown == [] and "couldn't find" in answer.lower()
    assert client.calls == []


def test_decomposer_parses_a_json_array():
    client = ScriptedClient(['Here you go: ["yoga mat", "water bottle"]'])
    assert QueryDecomposer(client).decompose("yoga mat and water bottle") == [
        "yoga mat",
        "water bottle",
    ]


def test_decomposer_caps_the_number_of_sub_queries():
    client = ScriptedClient(['["a","b","c","d","e","f"]'])
    assert len(QueryDecomposer(client).decompose("many things")) == 4


@pytest.mark.parametrize("reply", ["not json at all", "[]", "{'a':1}", ""])
def test_decomposer_falls_back_to_the_query_on_bad_output(reply):
    client = ScriptedClient([reply])
    assert QueryDecomposer(client).decompose("red shoes", fallback="red shoes") == ["red shoes"]


def test_decomposer_falls_back_when_the_llm_errors():
    from retailmind.generate.llm import LLMError

    class Broken:
        def complete(self, *a, **k):
            raise LLMError("down")

    assert QueryDecomposer(Broken()).decompose("red shoes") == ["red shoes"]
