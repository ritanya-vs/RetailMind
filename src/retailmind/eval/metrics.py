"""Ranking metrics computed with ranx against the ESCI judgments.

Graded nDCG uses the label gains (E 1, S 0.1, C 0.01, I 0). The other metrics are binary and treat
only Exact matches as relevant. Products that were never judged for a query count as not relevant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ranx import Qrels, Run, evaluate

Results = dict[str, list[tuple[str, float]]]  # query_id -> [(product_id, score), ...] best first

# ranx stores relevance scores as integers, which would turn the Substitute gain 0.1 into 0.
# Scaling every gain by 100 keeps the E : S : C : I ratios exactly (100 : 10 : 1 : 0).
GAIN_SCALE = 100

METRIC_NAMES = ("ndcg@10", "ndcg@10_exact", "mrr@10", "recall@100")
_EMPTY_PLACEHOLDER = "__no_results__"


def load_split(processed_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (queries, qrels) for split 'dev' or 'test' from the prepared files."""
    processed_dir = Path(processed_dir)
    queries = pd.read_parquet(processed_dir / "queries.parquet")
    queries = queries[queries["split"] == split].reset_index(drop=True)
    if queries.empty:
        raise ValueError(f"no queries for split {split!r}")
    qrels = pd.read_csv(
        processed_dir / "qrels.tsv", sep="\t", dtype={"query_id": str, "product_id": str}
    )
    queries["query_id"] = queries["query_id"].astype(str)
    return queries, qrels[qrels["query_id"].isin(queries["query_id"])].reset_index(drop=True)


def _qrels(frame: pd.DataFrame, *, exact_only: bool) -> Qrels:
    frame = frame[frame["label"] == "E"] if exact_only else frame[frame["gain"] > 0]
    gains: dict[str, dict[str, int]] = {}
    for query_id, product_id, gain in zip(
        frame["query_id"], frame["product_id"], frame["gain"], strict=True
    ):
        gains.setdefault(query_id, {})[product_id] = 1 if exact_only else round(gain * GAIN_SCALE)
    return Qrels.from_dict(gains)


def _run(results: Results, query_ids: list[str]) -> Run:
    scored = {}
    for query_id in query_ids:
        hits = results.get(query_id) or [(_EMPTY_PLACEHOLDER, 0.0)]
        scored[query_id] = {product_id: float(score) for product_id, score in hits}
    return Run.from_dict(scored)


def evaluate_results(results: Results, qrels: pd.DataFrame) -> tuple[dict, dict]:
    """Score ranked results. Returns (mean per metric, per-query scores per metric).

    Queries missing from results score 0, so a system cannot gain by returning nothing.
    """
    query_ids = sorted(qrels["query_id"].unique())
    run = _run(results, query_ids)
    graded, exact = _qrels(qrels, exact_only=False), _qrels(qrels, exact_only=True)

    per_query = {
        "ndcg@10": evaluate(graded, run, "ndcg@10", return_mean=False),
        "ndcg@10_exact": evaluate(exact, run, "ndcg@10", return_mean=False),
        "mrr@10": evaluate(exact, run, "mrr@10", return_mean=False),
        "recall@100": evaluate(exact, run, "recall@100", return_mean=False),
    }
    # ranx returns one score per query in sorted query-id order; store them by id
    order = query_ids
    means = {name: float(np.mean(values)) for name, values in per_query.items()}
    by_query = {
        name: {qid: float(v) for qid, v in zip(order, values, strict=True)}
        for name, values in per_query.items()
    }
    return means, by_query
