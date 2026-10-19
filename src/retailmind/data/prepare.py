"""Turn the raw ESCI files into the small, clean files the rest of the project reads.

Outputs, all in the processed directory:
    products.parquet       product_id, title, brand, color, text   (the search corpus)
    queries.parquet        query_id, query, split (dev or test), n_judged, n_exact
    qrels.tsv              query_id, product_id, label, gain       (relevance judgments)
    PREPARE_SUMMARY.json   counts and settings, for checking the run

Dev queries are sampled from the ESCI train split and test queries from the ESCI test split, so
tuning on dev never touches the official test queries.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from retailmind.config import DataConfig
from retailmind.data.esci import ESCI_FILES

EXAMPLES_FILE = ESCI_FILES[0].name
PRODUCTS_FILE = ESCI_FILES[1].name

# Graded gains for ndcg. The code released with ESCI swapped the S and C gains; this is the
# order the paper describes: Exact > Substitute > Complement > Irrelevant.
LABEL_GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

_TAG = re.compile(r"<[^>]*>")
_SPACE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Remove HTML, entities, emoji and control characters, and collapse whitespace."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = html.unescape(str(value))
    text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if not unicodedata.category(ch).startswith("C") or ch.isspace()
    )
    text = "".join(ch for ch in text if unicodedata.category(ch) != "So")
    return _SPACE.sub(" ", text).strip()


def build_text(
    title: str, brand: str, color: str, bullets: str, description: str, max_chars: int
) -> str:
    """Join the cleaned fields into one searchable string, cut at a word boundary."""
    text = " ".join(part for part in (title, brand, color, bullets, description) if part)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if text[max_chars] != " " and " " in cut:  # do not end in the middle of a word
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _read_examples(raw_dir: Path, cfg: DataConfig) -> pd.DataFrame:
    filters = [("product_locale", "=", cfg.locale)]
    if cfg.small_version:
        filters.append(("small_version", "=", 1))
    frame = pd.read_parquet(
        raw_dir / EXAMPLES_FILE,
        columns=["query_id", "query", "product_id", "esci_label", "split", "small_version"],
        filters=filters,
    )
    unknown = sorted(set(frame["esci_label"]) - set(LABEL_GAIN))
    if unknown:
        raise ValueError(f"unexpected ESCI labels: {unknown}")
    return frame.drop(columns="small_version")


def select_queries(examples: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Sample the dev queries from the train split and the test queries from the test split."""
    rng = np.random.default_rng(cfg.seed)
    parts = []
    for source_split, name, n in (("train", "dev", cfg.n_dev), ("test", "test", cfg.n_test)):
        pool = (
            examples.loc[examples["split"] == source_split, ["query_id", "query"]]
            .drop_duplicates("query_id")
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        if len(pool) < n:
            raise ValueError(
                f"need {n} {name} queries but the {source_split} split has {len(pool)}"
            )
        chosen = pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))].copy()
        chosen["split"] = name
        parts.append(chosen)
    return pd.concat(parts, ignore_index=True)


def _load_products(raw_dir: Path, wanted: set, locale: str, batch_size: int) -> pd.DataFrame:
    """Read the large products file in batches, keeping only the wanted products."""
    columns = [
        "product_id",
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
        "product_locale",
    ]
    kept = []
    with open(raw_dir / PRODUCTS_FILE, "rb") as source:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            frame = frame[(frame["product_locale"] == locale) & frame["product_id"].isin(wanted)]
            if len(frame):
                kept.append(frame.drop(columns="product_locale"))
    if not kept:
        return pd.DataFrame(columns=columns[:-1])
    return pd.concat(kept, ignore_index=True)


def _clean_products(raw: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    rows = []
    for rec in raw.itertuples(index=False):
        title = clean_text(rec.product_title)
        brand = clean_text(rec.product_brand)
        color = clean_text(rec.product_color)
        bullets = clean_text(rec.product_bullet_point)
        description = clean_text(rec.product_description)
        rows.append(
            {
                "product_id": rec.product_id,
                "title": title,
                "brand": brand,
                "color": color,
                "text": build_text(title, brand, color, bullets, description, max_chars),
            }
        )
    return pd.DataFrame(rows, columns=["product_id", "title", "brand", "color", "text"])


def prepare(
    raw_dir: str | Path,
    out_dir: str | Path,
    cfg: DataConfig,
    *,
    batch_size: int = 100_000,
) -> dict:
    """Build the processed files from the raw ESCI files and return a summary."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = _read_examples(raw_dir, cfg)
    queries = select_queries(examples, cfg)

    qrels = examples.merge(queries[["query_id", "split"]], on="query_id", suffixes=("_esci", ""))
    qrels = qrels.drop(columns=["query", "split_esci"]).rename(columns={"esci_label": "label"})
    if qrels.duplicated(["query_id", "product_id"]).any():
        raise ValueError("duplicate (query_id, product_id) judgments in the examples file")
    qrels["gain"] = qrels["label"].map(LABEL_GAIN)
    judged = set(qrels["product_id"])

    # Distractors: judged products of queries that were not sampled, so the corpus has
    # realistic competitors for every query. Chosen with the same seed.
    pool = sorted(set(examples["product_id"]) - judged)
    n_extra = min(cfg.corpus_distractors, len(pool))
    rng = np.random.default_rng(cfg.seed + 1)
    extra = set(rng.choice(pool, size=n_extra, replace=False).tolist()) if n_extra else set()

    raw_products = _load_products(raw_dir, judged | extra, cfg.locale, batch_size)
    raw_products = raw_products.drop_duplicates("product_id")
    missing = judged - set(raw_products["product_id"])
    if missing:
        raise ValueError(f"{len(missing)} judged products are missing from the products file")
    products = _clean_products(raw_products, cfg.text_max_chars)
    products = products.sort_values("product_id").reset_index(drop=True)

    counts = qrels.groupby("query_id")["label"].agg(
        n_judged="size", n_exact=lambda labels: int((labels == "E").sum())
    )
    queries = queries.merge(counts, left_on="query_id", right_index=True, how="left")
    queries = queries.sort_values("query_id").reset_index(drop=True)
    qrels = qrels[["query_id", "product_id", "label", "gain"]]
    qrels = qrels.sort_values(["query_id", "product_id"]).reset_index(drop=True)

    products.to_parquet(out_dir / "products.parquet", index=False)
    queries.to_parquet(out_dir / "queries.parquet", index=False)
    qrels.to_csv(out_dir / "qrels.tsv", sep="\t", index=False)

    summary = {
        "locale": cfg.locale,
        "small_version": cfg.small_version,
        "seed": cfg.seed,
        "n_dev_queries": int((queries["split"] == "dev").sum()),
        "n_test_queries": int((queries["split"] == "test").sum()),
        "n_qrels": len(qrels),
        "label_counts": {k: int(v) for k, v in qrels["label"].value_counts().sort_index().items()},
        "n_products": len(products),
        "n_judged_products": len(judged),
        "n_distractor_products": len(products) - len(judged),
        "queries_without_exact": int((queries["n_exact"] == 0).sum()),
        "products_without_text": int((products["text"] == "").sum()),
        "mean_text_chars": round(float(products["text"].str.len().mean()), 1),
    }
    (out_dir / "PREPARE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
