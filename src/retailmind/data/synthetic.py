"""Synthetic price, delivery and stock values for every product.

ESCI has no price, delivery or availability data, so these three fields are generated. They are
NOT real: they exist so that price, delivery and stock constraints can be tested. Each value
depends only on the product id and the seed, so it does not change when the corpus changes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

DELIVERY_DAYS = (1, 2, 3, 4, 5, 7, 10)
DELIVERY_WEIGHTS = (0.15, 0.25, 0.25, 0.15, 0.10, 0.07, 0.03)
MEDIAN_PRICE = 25.0
PRICE_SPREAD = 0.9  # sigma of the log-normal price
IN_STOCK_RATE = 0.9


def _rng(product_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{product_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def synthetic_attributes(product_ids: Sequence[str], seed: int) -> pd.DataFrame:
    """Return a frame with product_id, price (USD), delivery_days and in_stock."""
    rows = []
    for product_id in product_ids:
        rng = _rng(product_id, seed)
        price = float(np.clip(np.exp(rng.normal(np.log(MEDIAN_PRICE), PRICE_SPREAD)), 1, 800))
        rows.append(
            {
                "product_id": product_id,
                "price": round(price, 2),
                "delivery_days": int(rng.choice(DELIVERY_DAYS, p=DELIVERY_WEIGHTS)),
                "in_stock": bool(rng.random() < IN_STOCK_RATE),
            }
        )
    return pd.DataFrame(rows, columns=["product_id", "price", "delivery_days", "in_stock"])
