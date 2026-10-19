"""Build the processed dataset (products, queries, qrels) from the raw ESCI files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed
"""

import argparse
import sys
from pathlib import Path

from retailmind.config import load_config
from retailmind.data.prepare import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", help="override paths.raw_dir from the config")
    parser.add_argument("--out-dir", help="override paths.processed_dir from the config")
    args = parser.parse_args(argv)

    cfg = load_config()
    raw_dir = Path(args.raw_dir) if args.raw_dir else cfg.resolve(cfg.paths.raw_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cfg.resolve(cfg.paths.processed_dir)

    try:
        summary = prepare(raw_dir, out_dir, cfg.data)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}", file=sys.stderr)
        return 1

    print(f"Wrote products.parquet, queries.parquet, qrels.tsv to {out_dir}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if summary["queries_without_exact"]:
        print("Warning: some queries have no Exact match; check PREPARE_SUMMARY.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
