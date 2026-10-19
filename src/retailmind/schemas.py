"""Pydantic models shared by all pipeline stages.

Stages exchange only these types, so no stage needs to import another stage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Constraints(BaseModel):
    """Hard constraints extracted from a query. Empty or null means 'not specified'."""

    brand: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    size: list[str] = Field(default_factory=list)
    material: list[str] = Field(default_factory=list)
    price_min: float | None = Field(default=None, ge=0)
    price_max: float | None = Field(default=None, ge=0)
    delivery_max_days: int | None = Field(default=None, ge=0)
    in_stock_only: bool = False
    exclude_terms: list[str] = Field(default_factory=list)  # words the product text must not have

    @model_validator(mode="after")
    def _price_range_is_ordered(self) -> Constraints:
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min must not exceed price_max")
        return self

    def active(self) -> list[str]:
        """Names of the constraints that are actually set, e.g. ['color', 'price']."""
        names = [n for n in ("brand", "color", "size", "material") if getattr(self, n)]
        if self.price_min is not None or self.price_max is not None:
            names.append("price")
        if self.delivery_max_days is not None:
            names.append("delivery")
        if self.in_stock_only:
            names.append("in_stock")
        if self.exclude_terms:
            names.append("exclude")
        return names

    def is_empty(self) -> bool:
        return not self.active()


class ParsedQuery(BaseModel):
    """Output of the query parser."""

    query_id: str
    raw: str
    product_terms: str
    constraints: Constraints = Field(default_factory=Constraints)
    soft_preferences: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)


class StageScores(BaseModel):
    """Score of one product at each stage. None means the stage was not run."""

    bm25: float | None = None
    dense: float | None = None
    rrf: float | None = None
    rerank: float | None = None


class SearchHit(BaseModel):
    """One ranked product in a response."""

    product_id: str
    rank: int = Field(ge=1)
    scores: StageScores = Field(default_factory=StageScores)
    passed_filters: bool = True
    violations: list[str] = Field(default_factory=list)


class PipelineResponse(BaseModel):
    """Output of pipeline.search, shown in the demo and logged per evaluation run."""

    query_id: str
    parsed: ParsedQuery | None = None
    hits: list[SearchHit] = Field(default_factory=list)
    answer: str | None = None
    iterations: int = Field(default=1, ge=0)
    trace: list[str] = Field(default_factory=list)
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
