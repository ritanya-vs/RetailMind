"""Measure how well parsed constraints agree with the ESCI judgments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import pandas as pd

from retailmind.query.lexicon import tokens
from retailmind.schemas import ParsedQuery

LABELS = ("E", "S", "C", "I")
KINDS = ("brand", "color", "size", "material", "exclude_terms")


def product_info(products: pd.DataFrame) -> dict[str, dict]:
    """Normalized brand, color and text of each product, ready for matching."""
    return {
        row.product_id: {
            "brand": " ".join(tokens(row.brand)),
            "color_tokens": set(tokens(row.color)),
            "text": f" {' '.join(tokens(row.text))} ",
        }
        for row in products.itertuples()
    }


def _in_text(phrase: str, product: Mapping) -> bool:
    return f" {' '.join(tokens(phrase))} " in product["text"]


def satisfies(kind: str, wanted: Sequence[str], product: Mapping, use_text: bool = False) -> bool:
    """True when the product has at least one wanted value (for exclude_terms: any term).

    Brand and color are first checked against the product's own field. With use_text the product
    text also counts, which catches products whose field is empty or phrased differently.
    """
    if kind == "brand":
        if product["brand"] in {" ".join(tokens(b)) for b in wanted}:
            return True
    elif kind == "color":
        if any(set(tokens(c)) <= product["color_tokens"] for c in wanted):
            return True
    return (use_text or kind not in ("brand", "color")) and any(
        _in_text(w, product) for w in wanted
    )


def agreement(
    parsed: Mapping[str, ParsedQuery],
    judged: Mapping[str, Sequence[tuple[str, str]]],
    info: Mapping[str, Mapping],
    kind: str,
    use_text: bool = False,
) -> dict:
    """Share of judged products satisfying the parsed constraint, per ESCI label."""
    counts = {label: [0, 0] for label in LABELS}
    queries = [qid for qid, p in parsed.items() if getattr(p.constraints, kind)]
    for qid in queries:
        wanted = getattr(parsed[qid].constraints, kind)
        for product_id, label in judged.get(qid, ()):
            counts[label][1] += 1
            counts[label][0] += satisfies(kind, wanted, info[product_id], use_text)
    rates = {name: (m / t if t else None) for name, (m, t) in counts.items()}
    return {"queries": queries, "counts": counts, "rates": rates}


def group_judgments(qrels: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for qid, pid, label in zip(qrels["query_id"], qrels["product_id"], qrels["label"], strict=True):
        grouped[qid].append((pid, label))
    return grouped
