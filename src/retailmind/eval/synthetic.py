"""SC-100: a synthetic constrained query set with ground truth defined by construction.

ESCI queries rarely state price, delivery or stock limits, so they cannot show whether a system
respects such constraints. Each SC query takes an ESCI query and appends constraint phrases such as
"under $40 delivered in 2 days in stock". A product is relevant only if ESCI judged it relevant AND
its synthetic attributes meet every stated constraint. Limits are chosen from the attributes of
the query's Exact products so that some Exact products pass and some fail, which makes the
constraint matter. The attributes are synthetic (see retailmind.data.synthetic).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from retailmind.data.prepare import LABEL_GAIN
from retailmind.kg.graph import ProductKG
from retailmind.schemas import Constraints

TEMPLATES = (
    ("price",),
    ("delivery",),
    ("price", "delivery"),
    ("price", "in_stock"),
    ("price", "delivery", "in_stock"),
)
PRICE_PHRASES = ("under ${x}", "less than ${x}", "below ${x}")
DELIVERY_PHRASES = ("delivered in {d} days", "arrives within {d} days", "within {d} days delivery")
STOCK_PHRASES = ("in stock", "available now")
MIN_EXACT = 3
ATTEMPTS = 25


def _meets(constraints: Constraints, price: float, days: int, in_stock: bool) -> bool:
    if constraints.price_max is not None and price > constraints.price_max:
        return False
    if constraints.delivery_max_days is not None and days > constraints.delivery_max_days:
        return False
    return not (constraints.in_stock_only and not in_stock)


def _phrases(kinds: tuple[str, ...], constraints: Constraints, rng: np.random.Generator) -> str:
    parts = []
    for kind in kinds:
        if kind == "price":
            template = PRICE_PHRASES[rng.integers(len(PRICE_PHRASES))]
            parts.append(template.format(x=f"{constraints.price_max:g}"))
        elif kind == "delivery":
            days = constraints.delivery_max_days
            parts.append("next day delivery" if days == 1 else _delivery(days, rng))
        else:
            parts.append(STOCK_PHRASES[rng.integers(len(STOCK_PHRASES))])
    return " ".join(parts[i] for i in rng.permutation(len(parts)))


def _delivery(days: int, rng: np.random.Generator) -> str:
    return DELIVERY_PHRASES[rng.integers(len(DELIVERY_PHRASES))].format(d=days)


def _draw_constraints(
    kinds: tuple[str, ...], exact: pd.DataFrame, rng: np.random.Generator
) -> Constraints | None:
    """Pick limits so that at least one Exact product passes and at least one fails."""
    for _ in range(ATTEMPTS):
        constraints = Constraints(
            price_max=float(math.ceil(rng.choice(exact["price"]))) if "price" in kinds else None,
            delivery_max_days=int(rng.choice(exact["delivery_days"]))
            if "delivery" in kinds
            else None,
            in_stock_only="in_stock" in kinds,
        )
        passing = sum(
            _meets(constraints, row.price, row.delivery_days, row.in_stock)
            for row in exact.itertuples()
        )
        if 0 < passing < len(exact):
            return constraints
    return None


def make_sc_set(
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    attributes: pd.DataFrame,
    *,
    n: int = 100,
    seed: int = 42,
    prefix: str = "sc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build n constrained queries from queries (query_id, query) and their qrels."""
    rng = np.random.default_rng(seed)
    attrs = attributes.set_index("product_id")
    judged = {qid: group for qid, group in qrels.groupby("query_id")}
    order = rng.permutation(len(queries))
    rows, new_qrels = [], []
    for position in order:
        if len(rows) == n:
            break
        base = queries.iloc[position]
        group = judged.get(base["query_id"])
        if group is None:
            continue
        exact_ids = group.loc[group["label"] == "E", "product_id"].tolist()
        if len(exact_ids) < MIN_EXACT or not set(group["product_id"]) <= set(attrs.index):
            continue
        kinds = TEMPLATES[len(rows) % len(TEMPLATES)]
        exact = attrs.loc[exact_ids].reset_index()
        constraints = _draw_constraints(kinds, exact, rng)
        if constraints is None:
            continue
        query_id = f"{prefix}_{len(rows):03d}"
        text = f"{base['query']} {_phrases(kinds, constraints, rng)}"
        rows.append(
            {
                "query_id": query_id,
                "query": text,
                "base_query_id": base["query_id"],
                "template": "+".join(kinds),
                "constraints": constraints.model_dump_json(),
            }
        )
        for product_id, label in zip(group["product_id"], group["label"], strict=True):
            record = attrs.loc[product_id]
            ok = _meets(
                constraints, record["price"], int(record["delivery_days"]), bool(record["in_stock"])
            )
            final = label if ok else "I"
            new_qrels.append(
                {
                    "query_id": query_id,
                    "product_id": product_id,
                    "label": final,
                    "gain": LABEL_GAIN[final],
                }
            )
    if len(rows) < n:
        raise ValueError(f"could only build {len(rows)} of {n} constrained queries")
    return pd.DataFrame(rows), pd.DataFrame(new_qrels)


def load_sc(
    processed_dir: str | Path, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Constraints]]:
    """Return (queries, qrels, true constraints by query id) of the SC set for a split."""
    processed_dir = Path(processed_dir)
    queries = pd.read_parquet(processed_dir / f"sc_{split}_queries.parquet")
    qrels = pd.read_csv(
        processed_dir / f"sc_{split}_qrels.tsv",
        sep="\t",
        dtype={"query_id": str, "product_id": str},
    )
    truth = {
        qid: Constraints.model_validate(json.loads(raw))
        for qid, raw in zip(queries["query_id"], queries["constraints"], strict=True)
    }
    return queries, qrels, truth


def violation_rates(
    results: dict[str, list[tuple[str, float]]],
    truth: dict[str, Constraints],
    kg: ProductKG,
    k: int = 10,
) -> dict[str, float]:
    """How often the top k break the real constraints.

    violation@k is the mean share of top-k products breaking at least one constraint.
    clean@k is the share of queries whose top k contain no violation at all.
    """
    enabled = {"price", "delivery", "in_stock"}
    shares, clean = [], []
    for qid, constraints in truth.items():
        top = [pid for pid, _ in results.get(qid, [])[:k] if kg.has(pid)]
        if not top:
            shares.append(0.0)
            clean.append(1.0)
            continue
        bad = sum(bool(kg.violations(pid, constraints, enabled)) for pid in top)
        shares.append(bad / len(top))
        clean.append(float(bad == 0))
    return {f"violation@{k}": float(np.mean(shares)), f"clean@{k}": float(np.mean(clean))}
