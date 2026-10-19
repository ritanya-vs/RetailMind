"""Cross-encoder re-ranking of the fused candidate list."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

Candidates = Sequence[tuple[str, float]]  # [(product_id, score), ...] best first


class PairScorer(Protocol):
    """Anything that scores (query, document) pairs, higher meaning more relevant."""

    def score(
        self, pairs: Sequence[tuple[str, str]], *, show_progress: bool = False
    ) -> np.ndarray: ...


class CrossEncoderScorer:
    """Wraps a sentence-transformers CrossEncoder. The model is loaded on first use."""

    def __init__(self, model_name: str, *, batch_size: int = 32, max_length: int = 256) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def score(
        self, pairs: Sequence[tuple[str, str]], *, show_progress: bool = False
    ) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import, load lazily

            self._model = CrossEncoder(self.model_name)
            self._model.max_length = self.max_length  # cap the pair length to keep CPU runs fast
        scores = self._model.predict(
            [list(pair) for pair in pairs],
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float64).reshape(-1)


class Reranker:
    """Re-scores the top candidates of each query and keeps the rest below them in place.

    Only the first top_k candidates are scored. The remaining ones keep their order and get
    scores below every re-scored candidate, so a ranking still contains the full candidate list
    and recall at depth 100 is unchanged. After re-ranking, scores are only meaningful for
    ordering within one query.
    """

    def __init__(self, scorer: PairScorer, texts: Mapping[str, str]) -> None:
        self.scorer = scorer
        self._texts = texts

    def rerank_batch(
        self,
        queries: Sequence[str],
        candidates: Sequence[Candidates],
        top_k: int,
        *,
        show_progress: bool = False,
    ) -> list[list[tuple[str, float]]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        pairs: list[tuple[str, str]] = []
        heads: list[Candidates] = []
        for query, ranked in zip(queries, candidates, strict=True):
            head = ranked[:top_k]
            heads.append(head)
            for product_id, _ in head:
                if product_id not in self._texts:
                    raise KeyError(f"no text for product {product_id!r}")
                pairs.append((query, self._texts[product_id]))
        scores = self.scorer.score(pairs, show_progress=show_progress) if pairs else []
        if len(scores) != len(pairs):
            raise ValueError("scorer returned a different number of scores than pairs")

        results: list[list[tuple[str, float]]] = []
        offset = 0
        for head, ranked in zip(heads, candidates, strict=True):
            head_scores = [float(s) for s in scores[offset : offset + len(head)]]
            offset += len(head)
            order = sorted(range(len(head)), key=lambda i: -head_scores[i])  # stable on ties
            rescored = [(head[i][0], head_scores[i]) for i in order]
            floor = min(head_scores, default=0.0)
            tail = [(pid, floor - 1.0 - j) for j, (pid, _) in enumerate(ranked[len(head) :])]
            results.append(rescored + tail)
        return results

    def rerank(self, query: str, candidates: Candidates, top_k: int) -> list[tuple[str, float]]:
        return self.rerank_batch([query], [candidates], top_k)[0]
