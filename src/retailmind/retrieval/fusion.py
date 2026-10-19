"""Reciprocal rank fusion of several ranked lists."""

from __future__ import annotations

from collections.abc import Sequence

Ranking = Sequence[tuple[str, float]]  # [(product_id, score), ...] best first


def rrf(
    rankings: Sequence[Ranking], *, k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked lists. A product's score is the sum of 1 / (k + rank) over the lists it is in.

    Only ranks are used, never the original scores, so BM25 and cosine scores need no scaling.
    Ties keep the order in which the products were first seen (earlier lists first).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    fused: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, (product_id, _score) in enumerate(ranking, start=1):
            if product_id in seen:  # a repeated product only counts at its best rank
                continue
            seen.add(product_id)
            fused[product_id] = fused.get(product_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(fused.items(), key=lambda item: -item[1])  # stable: ties keep first seen
    return ordered[:top_k] if top_k is not None else ordered
