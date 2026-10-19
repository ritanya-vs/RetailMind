"""Run the BM25 baseline (ablation A0) on the dev or test queries and score it.

Usage:
    python scripts/run_baseline.py --split dev
    python scripts/run_baseline.py --split test
"""

import argparse
import json
import sys
import time

import pandas as pd

from retailmind.config import load_config
from retailmind.eval.metrics import evaluate_results, load_split
from retailmind.retrieval.bm25 import BM25Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--k", type=int, help="results per query (default: retrieval.bm25_k)")
    args = parser.parse_args(argv)

    cfg = load_config()
    k = args.k or cfg.retrieval.bm25_k
    processed = cfg.resolve(cfg.paths.processed_dir)
    out_dir = cfg.resolve(cfg.paths.results_dir) / f"A0_{args.split}"

    try:
        products = pd.read_parquet(processed / "products.parquet")
        queries, qrels = load_split(processed, args.split)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun scripts/prepare_data.py first.", file=sys.stderr)
        return 1

    start = time.perf_counter()
    index = BM25Retriever(products["product_id"].tolist(), products["text"].tolist())
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    results = {
        qid: index.search(text, k)
        for qid, text in zip(queries["query_id"], queries["query"], strict=True)
    }
    search_seconds = time.perf_counter() - start

    means, by_query = evaluate_results(results, qrels)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run": "A0",
        "components": ["bm25"],
        "split": args.split,
        "n_queries": len(queries),
        "k": k,
        "metrics": means,
        "index_build_seconds": round(build_seconds, 2),
        "ms_per_query": round(1000 * search_seconds / len(queries), 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "per_query.json").write_text(json.dumps(by_query), encoding="utf-8")
    (out_dir / "run.json").write_text(
        json.dumps({q: dict(hits) for q, hits in results.items()}), encoding="utf-8"
    )

    print(f"A0 BM25 on {len(queries)} {args.split} queries (top {k})")
    for name, value in means.items():
        print(f"  {name:15s} {value:.4f}")
    print(f"  index build {build_seconds:.1f}s, {report['ms_per_query']} ms/query")
    print(f"Saved metrics.json, per_query.json, run.json in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
