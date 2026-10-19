"""Generate the synthetic price, delivery and stock values for every product in the corpus.

Usage:
    python scripts/build_attributes.py
"""

import sys

import pandas as pd

from retailmind.config import load_config
from retailmind.data.synthetic import synthetic_attributes


def main() -> int:
    cfg = load_config()
    processed = cfg.resolve(cfg.paths.processed_dir)
    try:
        products = pd.read_parquet(processed / "products.parquet", columns=["product_id"])
    except OSError as error:
        print(f"Failed: {error}\nRun scripts/prepare_data.py first.", file=sys.stderr)
        return 1
    attributes = synthetic_attributes(products["product_id"].tolist(), cfg.data.seed)
    attributes.to_parquet(processed / "attributes.parquet", index=False)
    print(f"Wrote synthetic attributes for {len(attributes):,} products (NOT real data):")
    print(attributes[["price", "delivery_days", "in_stock"]].describe().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
