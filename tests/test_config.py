from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from retailmind.config import (
    COMPONENTS,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    Config,
    load_ablations,
    load_config,
)


def _raw_default() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_default_config_loads_with_expected_values():
    cfg = load_config()
    assert cfg.data.seed == 42
    assert cfg.data.locale == "us"
    assert cfg.retrieval.rrf_k == 60
    assert cfg.filter.mode == "lenient"
    assert cfg.rerank.top_k == 50
    assert cfg.agent.max_iterations == 2
    assert cfg.agent.min_rerank_score is None
    assert cfg.llm.model is None


def test_invalid_filter_mode_is_rejected():
    raw = _raw_default()
    raw["filter"]["mode"] = "weird"
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_unknown_key_is_rejected():
    raw = _raw_default()
    raw["retrieval"]["rrf_kk"] = 10  # typo for rrf_k
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_non_positive_sample_size_is_rejected():
    raw = _raw_default()
    raw["data"]["n_test"] = 0
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_resolve_returns_absolute_path_under_repo_root():
    cfg = load_config()
    resolved = cfg.resolve(cfg.paths.raw_dir)
    assert resolved.is_absolute()
    assert resolved == REPO_ROOT / "data" / "raw"


def test_load_config_from_custom_path(tmp_path: Path):
    raw = _raw_default()
    raw["data"]["seed"] = 7
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(custom).data.seed == 7


def test_ablations_cover_a0_to_a6():
    runs = load_ablations()
    assert list(runs) == [f"A{i}" for i in range(7)]
    assert runs["A0"].components == ["bm25"]
    assert set(runs["A6"].components) == COMPONENTS


def test_ablation_with_unknown_component_is_rejected(tmp_path: Path):
    bad = tmp_path / "ablations.yaml"
    bad.write_text(
        yaml.safe_dump({"runs": {"X": {"description": "bad", "components": ["bm25", "magic"]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_ablations(bad)


def test_ablations_file_without_runs_is_rejected(tmp_path: Path):
    empty = tmp_path / "ablations.yaml"
    empty.write_text(yaml.safe_dump({"other": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ablations(empty)
