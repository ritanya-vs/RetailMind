"""Run an ablation up to re-ranking and score it.

A0 BM25, A1 dense, A2 BM25 + dense with RRF, A3 A2 + cross-encoder re-ranking.

Usage:
    python scripts/run_retrieval.py --run A2 --split dev
    python scripts/run_retrieval.py --run A3 --split dev
"""

import argparse
import sys
import time

import pandas as pd

from retailmind.config import load_ablations, load_config
from retailmind.eval.metrics import evaluate_results, load_split
from retailmind.eval.report import print_metrics, save_run
from retailmind.rerank.cross_encoder import CrossEncoderScorer, Reranker
from retailmind.retrieval.bm25 import BM25Retriever
from retailmind.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from retailmind.retrieval.fusion import rrf

RETRIEVAL_COMPONENTS = {"bm25", "dense"}
SUPPORTED_COMPONENTS = RETRIEVAL_COMPONENTS | {"rerank"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="ablation name from configs/ablations.yaml")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
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

    cfg = load_config()
    processed = cfg.resolve(cfg.paths.processed_dir)
    out_dir = cfg.resolve(cfg.paths.results_dir) / f"{args.run}_{args.split}"
    try:
        products = pd.read_parquet(processed / "products.parquet")
        queries, qrels = load_split(processed, args.split)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun scripts/prepare_data.py first.", file=sys.stderr)
        return 1
    ids = products["product_id"].tolist()
    texts = queries["query"].tolist()
    query_ids = queries["query_id"].tolist()
    depth = max(cfg.retrieval.bm25_k, cfg.retrieval.dense_k)

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
    means, by_query = evaluate_results(results, qrels)
    report = {
        "run": args.run,
        "components": sorted(components),
        "split": args.split,
        "n_queries": len(queries),
        "depth": depth,
        "rrf_k": cfg.retrieval.rrf_k if len(lists) > 1 else None,
        "metrics": means,
        "rerank_model": cfg.rerank.model if "rerank" in components else None,
        "rerank_top_k": cfg.rerank.top_k if "rerank" in components else None,
        "seconds_retrieval": round(retrieval_seconds, 2),
        "seconds_rerank": round(rerank_seconds, 2),
    }
    save_run(out_dir, report, by_query, results)
    title = f"{args.run} ({runs[args.run].description}) on {len(queries)} {args.split} queries"
    print_metrics(title, means)
    print(f"  {retrieval_seconds:.1f}s retrieval + {rerank_seconds:.1f}s re-ranking")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
