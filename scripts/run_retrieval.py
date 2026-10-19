"""Run an ablation from retrieval up to the constraint filter and score it.

A0 BM25, A1 dense, A2 BM25 + dense with RRF, A3 A2 + cross-encoder re-ranking,
A4 A3 + query parser and constraint filter.

Usage:
    python scripts/run_retrieval.py --run A3 --split dev
    python scripts/run_retrieval.py --run A4 --split dev
    python scripts/run_retrieval.py --run A3 --dataset sc --split dev      constrained queries
    python scripts/run_retrieval.py --run A4 --dataset sc --split dev --filter-mode strict
"""

import argparse
import json
import sys
import time

import pandas as pd

from retailmind.config import FILTER_CONSTRAINTS, load_ablations, load_config
from retailmind.eval.metrics import evaluate_results, load_split
from retailmind.eval.report import print_metrics, save_run
from retailmind.eval.synthetic import load_sc, violation_rates
from retailmind.kg.build import load_kg, load_lexicon, parse_queries
from retailmind.kg.filter import ConstraintFilter
from retailmind.rerank.cross_encoder import CrossEncoderScorer, Reranker
from retailmind.retrieval.bm25 import BM25Retriever
from retailmind.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from retailmind.retrieval.fusion import rrf

RETRIEVAL_COMPONENTS = {"bm25", "dense"}
SUPPORTED_COMPONENTS = RETRIEVAL_COMPONENTS | {"rerank", "parse", "filter"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="ablation name from configs/ablations.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--dataset", choices=["esci", "sc"], default="esci")
    parser.add_argument("--filter-mode", choices=["strict", "lenient"])
    parser.add_argument("--penalty", type=float, help="soft penalty (brand, color, material)")
    parser.add_argument("--hard-penalty", type=float, help="penalty for price, delivery, stock")
    parser.add_argument("--constraints", help="comma list, e.g. price,delivery,in_stock")
    args = parser.parse_args(argv)

    runs = load_ablations()
    if args.run not in runs:
        print(f"Unknown run {args.run!r}; choose from {sorted(runs)}", file=sys.stderr)
        return 1
    components = set(runs[args.run].components)
    if not components <= SUPPORTED_COMPONENTS:
        extra = sorted(components - SUPPORTED_COMPONENTS)
        print(f"{args.run} needs components not built yet: {extra}", file=sys.stderr)
        return 1
    if not components & RETRIEVAL_COMPONENTS:
        print(f"{args.run} has nothing to retrieve with (needs bm25 and/or dense)", file=sys.stderr)
        return 1
    if "filter" in components and "parse" not in components:
        print("the filter needs the parser (component 'parse')", file=sys.stderr)
        return 1

    cfg = load_config()
    filter_mode = args.filter_mode or cfg.filter.mode
    penalty = cfg.filter.penalty if args.penalty is None else args.penalty
    hard_penalty = cfg.filter.hard_penalty if args.hard_penalty is None else args.hard_penalty
    enabled = args.constraints.split(",") if args.constraints else list(cfg.filter.constraints)
    if set(enabled) - FILTER_CONSTRAINTS:
        print(f"unknown constraints: {sorted(set(enabled) - FILTER_CONSTRAINTS)}", file=sys.stderr)
        return 1

    processed = cfg.resolve(cfg.paths.processed_dir)
    name = f"{args.run}_{args.split}" if args.dataset == "esci" else f"{args.run}_sc_{args.split}"
    out_dir = cfg.resolve(cfg.paths.results_dir) / name
    try:
        products = pd.read_parquet(processed / "products.parquet")
        truth = {}
        if args.dataset == "esci":
            queries, qrels = load_split(processed, args.split)
        else:
            queries, qrels, truth = load_sc(processed, args.split)
        lexicon = load_lexicon(cfg) if components & {"parse", "filter"} or truth else None
        kg = load_kg(cfg, lexicon) if "filter" in components or truth else None
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun the earlier data scripts first.", file=sys.stderr)
        return 1
    ids = products["product_id"].tolist()
    query_ids = queries["query_id"].tolist()
    depth = max(cfg.retrieval.bm25_k, cfg.retrieval.dense_k)

    parsed = {}
    texts = queries["query"].tolist()
    if "parse" in components:
        parsed = parse_queries(lexicon, query_ids, texts)
        texts = [parsed[qid].product_terms for qid in query_ids]  # without price/delivery/stock

    start = time.perf_counter()
    lists = []  # one list of per-query rankings for each retriever that is switched on
    if "bm25" in components:
        bm25 = BM25Retriever(ids, products["text"].tolist())
        lists.append([bm25.search(t, cfg.retrieval.bm25_k) for t in texts])
    if "dense" in components:
        encoder = SentenceTransformerEncoder(cfg.retrieval.dense_model)
        try:
            dense = DenseRetriever.load(
                cfg.resolve(cfg.paths.index_dir) / "dense",
                encoder,
                expected_model=cfg.retrieval.dense_model,
                expected_ids=ids,
            )
        except (OSError, ValueError) as error:
            print(f"Failed: {error}\nRun scripts/build_dense_index.py first.", file=sys.stderr)
            return 1
        lists.append(dense.search_batch(texts, cfg.retrieval.dense_k))

    if len(lists) == 1:
        per_query = lists[0]
    else:
        per_query = [
            rrf(rankings, k=cfg.retrieval.rrf_k, top_k=depth)
            for rankings in zip(*lists, strict=True)
        ]
    retrieval_seconds = time.perf_counter() - start

    rerank_seconds = 0.0
    if "rerank" in components:
        start = time.perf_counter()
        scorer = CrossEncoderScorer(cfg.rerank.model)
        reranker = Reranker(scorer, dict(zip(ids, products["text"], strict=True)))
        per_query = reranker.rerank_batch(texts, per_query, cfg.rerank.top_k, show_progress=True)
        rerank_seconds = time.perf_counter() - start

    results = dict(zip(query_ids, per_query, strict=True))
    if "filter" in components:
        # keep the unfiltered ranking so filter settings can be tuned without re-ranking again
        out_dir.mkdir(parents=True, exist_ok=True)
        prefilter = {q: dict(hits) for q, hits in results.items()}
        (out_dir / "run_prefilter.json").write_text(json.dumps(prefilter), encoding="utf-8")
        flt = ConstraintFilter(
            kg, mode=filter_mode, penalty=penalty, hard_penalty=hard_penalty, enabled=enabled
        )
        results = {q: flt.apply(parsed[q].constraints, hits)[0] for q, hits in results.items()}

    means, by_query = evaluate_results(results, qrels)
    extra = violation_rates(results, truth, kg) if truth else {}
    report = {
        "run": args.run,
        "dataset": args.dataset,
        "components": sorted(components),
        "split": args.split,
        "n_queries": len(queries),
        "depth": depth,
        "rrf_k": cfg.retrieval.rrf_k if len(lists) > 1 else None,
        "metrics": means,
        "constraint_metrics": extra,
        "filter": {
            "mode": filter_mode,
            "penalty": penalty,
            "hard_penalty": hard_penalty,
            "constraints": enabled,
        }
        if "filter" in components
        else None,
        "n_queries_with_constraints": sum(not p.constraints.is_empty() for p in parsed.values()),
        "rerank_model": cfg.rerank.model if "rerank" in components else None,
        "rerank_top_k": cfg.rerank.top_k if "rerank" in components else None,
        "seconds_retrieval": round(retrieval_seconds, 2),
        "seconds_rerank": round(rerank_seconds, 2),
    }
    save_run(out_dir, report, by_query, results)
    title = f"{args.run} ({runs[args.run].description}) on {len(queries)} {name} queries"
    print_metrics(title, {**means, **extra})
    print(f"  {retrieval_seconds:.1f}s retrieval + {rerank_seconds:.1f}s re-ranking")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
