"""Build the brand and color vocabulary used by the query parser from the full products file.

Usage:
    python scripts/build_lexicon.py
"""

import argparse
import sys
import time

from retailmind.config import load_config
from retailmind.data.esci import ESCI_FILES
from retailmind.query.lexicon import build_lexicon_from_products


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-brand-count", type=int, default=5)
    parser.add_argument("--min-precision", type=float, default=0.3)
    parser.add_argument("--min-color-count", type=int, default=50)
    args = parser.parse_args(argv)

    cfg = load_config()
    products_file = cfg.resolve(cfg.paths.raw_dir) / ESCI_FILES[1].name
    out_path = cfg.resolve(cfg.paths.processed_dir) / "lexicon.json"
    if not products_file.exists():
        print(f"Failed: {products_file} not found. Run scripts/download_esci.py first.")
        return 1

    start = time.perf_counter()
    print(f"Reading brands, colors and titles for locale {cfg.data.locale!r} (about a minute)...")
    lexicon, n_products = build_lexicon_from_products(
        products_file,
        cfg.data.locale,
        min_brand_count=args.min_brand_count,
        min_precision=args.min_precision,
        min_color_count=args.min_color_count,
    )
    lexicon.save(out_path)
    print(
        f"{n_products:,} products -> {len(lexicon.brands):,} brands, {len(lexicon.colors):,} colors"
    )
    print(f"Saved {out_path} in {time.perf_counter() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
