"""Vocabularies of brands and colors learned from the product catalog.

Brands and colors come from the structured fields of the products file, so the parser only
recognizes names that products actually carry. A brand written as one word is kept only if that
word almost always means the brand: "nike" appears in titles nearly only for Nike products, while
"apple" or "women" appear in many titles for other brands and are dropped.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")

# Words that are never treated as a brand or color on their own.
GENERIC_WORDS = frozenset(
    """a an and are as at be by for from in is it its of on or the to with without women womens
    woman men mens man kids kid boys girls baby toddler unisex adult adults set pack case cover
    bag box new for pro plus mini max size small medium large big home office outdoor garden
    kitchen black white red blue green yellow pink purple orange brown gray grey silver gold
    other generic unbranded brand""".split()
)

# A learned color is kept only if one of its words is a real color name. Product records contain
# junk in the color field ("small", "star", "glass"), and those must not become query constraints.
BASE_COLORS = frozenset(
    """black white red blue green yellow orange purple pink brown gray grey silver gold navy beige
    tan maroon burgundy teal turquoise ivory khaki charcoal olive coral lavender magenta cream rose
    bronze copper mint peach lilac aqua cyan indigo violet mauve plum nude camo""".split()
)

# Materials are matched from a fixed list: product records have no material field to learn from.
MATERIALS = (
    "leather", "cotton", "wool", "silk", "linen", "denim", "velvet", "polyester", "nylon",
    "rubber", "silicone", "plastic", "glass", "ceramic", "porcelain", "stainless steel",
    "aluminum", "aluminium", "titanium", "brass", "copper", "bamboo", "wood", "wooden",
    "metal", "acrylic", "cast iron", "canvas", "suede", "fleece", "cashmere", "stone",
)  # fmt: skip


def tokens(text: object) -> list[str]:
    """Lowercase, strip accents, and split into letters and digits."""
    if text is None or (isinstance(text, float) and text != text):
        return []
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return _TOKEN.findall(folded.lower())


@dataclass(frozen=True)
class Lexicon:
    brands: frozenset[str] = field(default_factory=frozenset)
    colors: frozenset[str] = field(default_factory=frozenset)
    materials: frozenset[str] = field(default_factory=lambda: frozenset(MATERIALS))

    def save(self, path: str | Path) -> None:
        data = {
            "brands": sorted(self.brands),
            "colors": sorted(self.colors),
            "materials": sorted(self.materials),
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Lexicon:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            brands=frozenset(data["brands"]),
            colors=frozenset(data["colors"]),
            materials=frozenset(data["materials"]),
        )


class LexiconBuilder:
    """Accumulates counts over batches of products, then decides which names to keep."""

    def __init__(self) -> None:
        self.brand_counts: Counter[str] = Counter()
        self.color_counts: Counter[str] = Counter()
        self.title_frequency: Counter[str] = Counter()
        self.n_products = 0

    def update(
        self, brands: Iterable[object], colors: Iterable[object], titles: Iterable[object]
    ) -> None:
        for brand in brands:
            key = " ".join(tokens(brand))
            if key:
                self.brand_counts[key] += 1
        for color in colors:
            key = " ".join(tokens(color))
            if key:
                self.color_counts[key] += 1
        for title in titles:
            self.title_frequency.update(set(tokens(title)))
            self.n_products += 1

    def build(
        self, *, min_brand_count: int = 5, min_precision: float = 0.3, min_color_count: int = 50
    ) -> Lexicon:
        brands = set()
        for key, count in self.brand_counts.items():
            if count < min_brand_count:
                continue
            words = key.split()
            if all(w in GENERIC_WORDS or w.isdigit() for w in words):
                continue
            if len(words) == 1:
                word = words[0]
                if len(word) < 2 or word.isdigit():
                    continue
                if count / max(self.title_frequency[word], 1) < min_precision:
                    continue
            brands.add(key)
        colors = {
            key
            for key, count in self.color_counts.items()
            if count >= min_color_count
            and len(key.split()) <= 3
            and all(w.isalpha() and len(w) >= 3 for w in key.split())
            and any(w in BASE_COLORS for w in key.split())
        }
        return Lexicon(brands=frozenset(brands), colors=frozenset(colors))


def build_lexicon_from_products(
    products_path: str | Path,
    locale: str = "us",
    *,
    batch_size: int = 100_000,
    min_brand_count: int = 5,
    min_precision: float = 0.3,
    min_color_count: int = 50,
) -> tuple[Lexicon, int]:
    """Read the big products file in batches and build a lexicon. Returns (lexicon, n_products)."""
    import pyarrow.parquet as pq

    builder = LexiconBuilder()
    columns = ["product_brand", "product_color", "product_title", "product_locale"]
    with open(products_path, "rb") as source:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            frame = frame[frame["product_locale"] == locale]
            builder.update(frame["product_brand"], frame["product_color"], frame["product_title"])
    lexicon = builder.build(
        min_brand_count=min_brand_count,
        min_precision=min_precision,
        min_color_count=min_color_count,
    )
    return lexicon, builder.n_products
