"""Build the synthetic constrained query sets (SC) for dev and test.

Usage:
    python scripts/make_sc_set.py            100 queries per split
    python scripts/make_sc_set.py --n 60
"""

import argparse
import sys

import pandas as pd

from retailmind.config import load_config
from retailmind.eval.metrics import load_split
from retailmind.eval.synthetic import make_sc_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100, help="queries per split")
    args = parser.parse_args(argv)

    cfg = load_config()
    processed = cfg.resolve(cfg.paths.processed_dir)
    try:
        attributes = pd.read_parquet(processed / "attributes.parquet")
        for offset, split in enumerate(("dev", "test")):
            queries, qrels = load_split(processed, split)
            sc_queries, sc_qrels = make_sc_set(
                queries,
                qrels,
                attributes,
                n=args.n,
                seed=cfg.data.seed + offset,
                prefix=f"sc{split}",
            )
            sc_queries.to_parquet(processed / f"sc_{split}_queries.parquet", index=False)
            sc_qrels.to_csv(processed / f"sc_{split}_qrels.tsv", sep="\t", index=False)
            print(f"SC-{split}: {len(sc_queries)} queries, {len(sc_qrels)} judgments")
            print("  by template:", sc_queries["template"].value_counts().to_dict())
            for text in sc_queries["query"].head(5):
                print("  e.g.", text)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun scripts/build_attributes.py first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
