"""Embed every product with the configured sentence model and save a FAISS-searchable index.

Usage:
    python scripts/build_dense_index.py            build the index (skips if one already matches)
    python scripts/build_dense_index.py --force    rebuild even if an index exists
"""

import argparse
import sys
import time

import pandas as pd

from retailmind.config import load_config
from retailmind.retrieval.dense import DenseRetriever, SentenceTransformerEncoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild an existing index")
    args = parser.parse_args(argv)

    cfg = load_config()
    model = cfg.retrieval.dense_model
    processed = cfg.resolve(cfg.paths.processed_dir)
    index_dir = cfg.resolve(cfg.paths.index_dir) / "dense"
    encoder = SentenceTransformerEncoder(model)

    try:
        products = pd.read_parquet(processed / "products.parquet")
    except OSError as error:
        print(f"Failed: {error}\nRun scripts/prepare_data.py first.", file=sys.stderr)
        return 1
    ids = products["product_id"].tolist()

    if not args.force:
        try:
            DenseRetriever.load(index_dir, encoder, expected_model=model, expected_ids=ids)
            print(f"Dense index in {index_dir} is up to date. Use --force to rebuild.")
            return 0
        except (OSError, ValueError):
            pass  # missing or stale, so build it

    print(f"Embedding {len(ids):,} products with {model} (first run downloads the model)...")
    start = time.perf_counter()
    retriever = DenseRetriever.build(ids, products["text"].tolist(), encoder, show_progress=True)
    retriever.save(index_dir, model)
    print(f"Saved dense index to {index_dir} in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
