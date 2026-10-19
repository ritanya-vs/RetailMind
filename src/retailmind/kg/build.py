"""Load everything needed to build the knowledge graph and parse queries."""

from __future__ import annotations

import pandas as pd

from retailmind.config import Config
from retailmind.kg.graph import ProductKG
from retailmind.query.lexicon import Lexicon
from retailmind.query.parser import QueryParser
from retailmind.schemas import ParsedQuery


def load_lexicon(cfg: Config) -> Lexicon:
    return Lexicon.load(cfg.resolve(cfg.paths.processed_dir) / "lexicon.json")


def load_kg(cfg: Config, lexicon: Lexicon) -> ProductKG:
    """Build the graph from the processed products and the synthetic attributes."""
    processed = cfg.resolve(cfg.paths.processed_dir)
    products = pd.read_parquet(processed / "products.parquet")
    attributes = pd.read_parquet(processed / "attributes.parquet")
    return ProductKG.from_products(products, attributes, lexicon)


def parse_queries(
    lexicon: Lexicon, query_ids: list[str], texts: list[str]
) -> dict[str, ParsedQuery]:
    parser = QueryParser(lexicon)
    return {qid: parser.parse(text, qid) for qid, text in zip(query_ids, texts, strict=True)}
