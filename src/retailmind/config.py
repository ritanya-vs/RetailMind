"""Typed configuration loaded from the YAML files in configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# src/retailmind/config.py -> parents[0] = retailmind, [1] = src, [2] = repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_ABLATIONS_PATH = REPO_ROOT / "configs" / "ablations.yaml"

# Pipeline components that can be switched on or off for an ablation run.
COMPONENTS = frozenset(
    {"bm25", "dense", "parse", "decompose", "filter", "rerank", "agent", "generate"}
)


class _Strict(BaseModel):
    """Config models reject unknown keys, so a typo in a YAML file fails loudly."""

    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    raw_dir: str
    processed_dir: str
    index_dir: str
    results_dir: str
    cache_dir: str


class DataConfig(_Strict):
    locale: str = "us"
    small_version: bool = True
    n_dev: int = Field(gt=0)
    n_test: int = Field(gt=0)
    seed: int
    text_max_chars: int = Field(default=2000, gt=0)  # cap on the searchable text per product
    corpus_distractors: int = Field(default=10000, ge=0)  # extra products beyond the judged ones


class RetrievalConfig(_Strict):
    bm25_k: int = Field(gt=0)
    dense_model: str
    dense_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)


# Constraint kinds the filter can enforce. Size and exclusions are parsed but not enforced.
FILTER_CONSTRAINTS = frozenset({"price", "delivery", "in_stock", "brand", "color", "material"})


class FilterConfig(_Strict):
    mode: Literal["strict", "lenient"] = "lenient"
    penalty: float = Field(default=0.5, ge=0)  # lenient: score lost per brand/color/material miss
    hard_penalty: float = Field(
        default=100.0, ge=0
    )  # lenient: score lost per price/delivery/stock miss
    constraints: list[str] = Field(default_factory=lambda: sorted(FILTER_CONSTRAINTS))

    @field_validator("constraints")
    @classmethod
    def _known_constraints(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - FILTER_CONSTRAINTS)
        if unknown:
            raise ValueError(f"unknown constraints: {unknown}")
        return value


class RerankConfig(_Strict):
    model: str
    top_k: int = Field(gt=0)


class AgentConfig(_Strict):
    min_good_results: int = Field(gt=0)
    min_rerank_score: float | None = None
    max_iterations: int = Field(ge=0)


class LLMConfig(_Strict):
    # Any OpenAI-compatible chat endpoint: Ollama (local or tunnelled), Groq, etc.
    # base_url, model and the API key are normally set in .env so the ephemeral tunnel URL and the
    # key stay out of git. The env vars LLM_BASE_URL, LLM_MODEL and LLM_API_KEY override these.
    provider: str = "openai_compatible"
    model: str | None = None  # e.g. "llama3.1" for Ollama, or a Groq model id
    base_url: str | None = (
        None  # e.g. "https://<id>.ngrok-free.app/v1" or "http://localhost:11434/v1"
    )
    api_key_env: str = "LLM_API_KEY"  # which env var holds the key (Ollama ignores it)
    cache_dir: str = ".cache/llm"
    temperature: float = Field(default=0.0, ge=0)  # 0 keeps answers reproducible
    max_tokens: int = Field(default=512, gt=0)
    max_context_products: int = Field(default=5, gt=0)  # products shown to the LLM as evidence
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)  # transient tunnel/network errors are retried


class Config(_Strict):
    paths: PathsConfig
    data: DataConfig
    retrieval: RetrievalConfig
    filter: FilterConfig
    rerank: RerankConfig
    agent: AgentConfig
    llm: LLMConfig

    def resolve(self, relative: str | Path) -> Path:
        """Turn a repository-relative path from the config into an absolute path."""
        return REPO_ROOT / relative


class AblationRun(_Strict):
    """One evaluation run: a name for the components that are switched on."""

    description: str
    components: list[str]

    @field_validator("components")
    @classmethod
    def _known_components(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("components must not be empty")
        unknown = sorted(set(value) - COMPONENTS)
        if unknown:
            raise ValueError(f"unknown components: {unknown}")
        return value


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate configs/default.yaml (or another YAML file)."""
    return Config.model_validate(_read_yaml(Path(path) if path else DEFAULT_CONFIG_PATH))


def load_ablations(path: str | Path | None = None) -> dict[str, AblationRun]:
    """Load and validate configs/ablations.yaml into {run name: AblationRun}."""
    source = Path(path) if path else DEFAULT_ABLATIONS_PATH
    runs = _read_yaml(source).get("runs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError(f"{source} must contain a non-empty 'runs' mapping")
    return {name: AblationRun.model_validate(spec) for name, spec in runs.items()}
