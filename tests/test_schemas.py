import json

import pytest
from pydantic import ValidationError

from retailmind.schemas import (
    Constraints,
    ParsedQuery,
    PipelineResponse,
    SearchHit,
)

PARSED_QUERY_JSON = """
{
  "query_id": "q1",
  "raw": "black running shoes for men under $60 delivered in 3 days",
  "product_terms": "black running shoes men",
  "constraints": {
    "brand": [],
    "color": ["black"],
    "size": [],
    "material": [],
    "price_min": null,
    "price_max": 60.0,
    "delivery_max_days": 3,
    "in_stock_only": false
  },
  "soft_preferences": [],
  "sub_queries": ["black running shoes for men", "men's running shoes"]
}
"""


def test_parsed_query_loads_the_documented_example():
    parsed = ParsedQuery.model_validate_json(PARSED_QUERY_JSON)
    assert parsed.constraints.color == ["black"]
    assert parsed.constraints.price_max == 60.0
    assert parsed.constraints.delivery_max_days == 3
    assert parsed.constraints.active() == ["color", "price", "delivery"]
    assert len(parsed.sub_queries) == 2


def test_parsed_query_round_trips_through_json():
    parsed = ParsedQuery.model_validate_json(PARSED_QUERY_JSON)
    again = ParsedQuery.model_validate_json(parsed.model_dump_json())
    assert again == parsed


def test_empty_constraints_have_no_active_names():
    constraints = Constraints()
    assert constraints.is_empty()
    assert constraints.active() == []


def test_in_stock_flag_counts_as_active():
    assert Constraints(in_stock_only=True).active() == ["in_stock"]


def test_price_range_must_be_ordered():
    with pytest.raises(ValidationError):
        Constraints(price_min=50, price_max=10)


def test_negative_price_and_delivery_are_rejected():
    with pytest.raises(ValidationError):
        Constraints(price_max=-1)
    with pytest.raises(ValidationError):
        Constraints(delivery_max_days=-2)


def test_search_hit_rank_starts_at_one():
    assert SearchHit(product_id="p1", rank=1).passed_filters is True
    with pytest.raises(ValidationError):
        SearchHit(product_id="p1", rank=0)


def test_pipeline_response_defaults_and_json_round_trip():
    response = PipelineResponse(
        query_id="q1",
        hits=[SearchHit(product_id="p1", rank=1, scores={"bm25": 12.5, "rrf": 0.03})],
        trace=["retrieve", "fuse"],
        latency_ms={"retrieve": 41.0, "rerank": None},
    )
    assert response.parsed is None
    assert response.answer is None
    assert response.iterations == 1
    assert response.hits[0].scores.dense is None

    restored = PipelineResponse.model_validate(json.loads(response.model_dump_json()))
    assert restored == response


def test_exclude_terms_count_as_active():
    assert Constraints(exclude_terms=["needle"]).active() == ["exclude"]
