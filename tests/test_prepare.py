import json
from pathlib import Path

import pandas as pd
import pytest

from retailmind.config import DataConfig
from retailmind.data.prepare import (
    EXAMPLES_FILE,
    LABEL_GAIN,
    PRODUCTS_FILE,
    build_text,
    clean_text,
    prepare,
)


def _cfg(**overrides) -> DataConfig:
    values = dict(n_dev=3, n_test=2, seed=7, text_max_chars=200, corpus_distractors=4)
    values.update(overrides)
    return DataConfig(**values)


def _write_raw(raw: Path, drop_product: str | None = None) -> None:
    """Fake ESCI: 8 train and 6 test US queries with 3 products each, plus rows to be ignored."""
    raw.mkdir(parents=True, exist_ok=True)
    labels = ["E", "S", "I"]
    examples, products = [], []
    for q in range(14):
        split = "train" if q < 8 else "test"
        for j in range(3):
            pid = f"P{q:02d}{j}"
            examples.append(
                dict(example_id=len(examples), query=f"query {q}", query_id=q, product_id=pid,
                     product_locale="us", esci_label=labels[j], small_version=1,
                     large_version=1, split=split)
            )
            products.append(
                dict(product_id=pid, product_title=f"Title <b>{pid}</b> &amp; co",
                     product_description=None, product_bullet_point="a\nb",
                     product_brand="Brand", product_color=None, product_locale="us")
            )
    # ignored: a Japanese row and a US row outside the small version
    examples.append(dict(example_id=900, query="jp", query_id=900, product_id="JP1",
                         product_locale="jp", esci_label="E", small_version=1,
                         large_version=1, split="train"))
    examples.append(dict(example_id=901, query="big", query_id=901, product_id="BIG1",
                         product_locale="us", esci_label="E", small_version=0,
                         large_version=1, split="train"))
    products.append(dict(product_id="JP1", product_title="jp", product_description=None,
                         product_bullet_point=None, product_brand=None, product_color=None,
                         product_locale="jp"))
    products.append(dict(product_id="BIG1", product_title="big", product_description=None,
                         product_bullet_point=None, product_brand=None, product_color=None,
                         product_locale="us"))
    products = [p for p in products if p["product_id"] != drop_product]
    pd.DataFrame(examples).to_parquet(raw / EXAMPLES_FILE)
    pd.DataFrame(products).to_parquet(raw / PRODUCTS_FILE)


def test_clean_text_removes_html_entities_symbols_and_extra_space():
    raw = "<p>Great &amp; cheap&#11088;</p>\n\n  now\tonly\x00!"
    assert clean_text(raw) == "Great & cheap now only!"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_build_text_joins_nonempty_parts_and_cuts_at_a_word():
    assert build_text("Shoe", "", "red", "", "soft", 100) == "Shoe red soft"
    assert build_text("alpha beta gamma", "", "", "", "", 12) == "alpha beta"


def test_prepare_writes_consistent_files(tmp_path: Path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_raw(raw)

    summary = prepare(raw, out, _cfg())

    queries = pd.read_parquet(out / "queries.parquet")
    products = pd.read_parquet(out / "products.parquet")
    qrels = pd.read_csv(out / "qrels.tsv", sep="\t", dtype={"product_id": str})
    assert (queries["split"] == "dev").sum() == 3 and (queries["split"] == "test").sum() == 2
    assert set(queries.loc[queries["split"] == "dev", "query_id"]) <= set(range(8))
    assert set(queries.loc[queries["split"] == "test", "query_id"]) <= set(range(8, 14))
    assert set(qrels["query_id"]) == set(queries["query_id"])
    assert set(qrels["product_id"]) <= set(products["product_id"])
    assert len(qrels) == 15
    assert (qrels["gain"] == qrels["label"].map(LABEL_GAIN)).all()
    assert (queries["n_judged"] == 3).all() and (queries["n_exact"] == 1).all()
    assert summary["n_judged_products"] == 15
    assert summary["n_distractor_products"] == 4
    assert summary["queries_without_exact"] == 0
    assert not products["product_id"].isin(["JP1", "BIG1"]).any()
    assert products["title"].str.contains("<").sum() == 0
    assert json.loads((out / "PREPARE_SUMMARY.json").read_text()) == summary


def test_prepare_is_deterministic(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_raw(raw)
    prepare(raw, tmp_path / "a", _cfg())
    prepare(raw, tmp_path / "b", _cfg())
    for name in ("products.parquet", "queries.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(tmp_path / "a" / name), pd.read_parquet(tmp_path / "b" / name)
        )
    assert (tmp_path / "a" / "qrels.tsv").read_text() == (tmp_path / "b" / "qrels.tsv").read_text()


def test_distractors_are_capped_by_what_is_available(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_raw(raw)
    summary = prepare(raw, tmp_path / "out", _cfg(corpus_distractors=1000))
    assert summary["n_distractor_products"] == 42 - 15  # 14 queries x 3 products, minus judged


def test_too_few_queries_fails(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_raw(raw)
    with pytest.raises(ValueError, match="need 50 dev queries"):
        prepare(raw, tmp_path / "out", _cfg(n_dev=50))


def test_judged_product_missing_from_products_file_fails(tmp_path: Path):
    raw = tmp_path / "raw"
    _write_raw(raw, drop_product="P000")
    # every query is sampled, so the dropped product is certain to be judged
    with pytest.raises(ValueError, match="missing from the products file"):
        prepare(raw, tmp_path / "out", _cfg(n_dev=8, n_test=6))
