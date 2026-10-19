"""Dense retrieval: sentence embeddings searched with a FAISS inner-product index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "product_ids.json"
META_FILE = "meta.json"


class Encoder(Protocol):
    """Anything that turns texts into a (n, dim) float array."""

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Wraps a sentence-transformers model. The model is loaded on first use."""

    def __init__(self, model_name: str, *, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, load lazily

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


def ids_fingerprint(product_ids: Sequence[str]) -> str:
    """Short hash of the ordered product ids, to detect an index built for a different corpus."""
    return hashlib.sha256("\n".join(product_ids).encode("utf-8")).hexdigest()[:16]


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class DenseRetriever:
    """Exact cosine search over unit-length embeddings."""

    def __init__(
        self,
        product_ids: Sequence[str],
        embeddings: np.ndarray,
        encoder: Encoder,
        *,
        normalize: bool = True,
    ) -> None:
        if embeddings.ndim != 2 or len(product_ids) != len(embeddings):
            raise ValueError("embeddings must be a (n_products, dim) array matching product_ids")
        if len(product_ids) == 0:
            raise ValueError("cannot build an index over an empty corpus")
        self.product_ids = list(product_ids)
        self.encoder = encoder
        self._embeddings = (
            _unit_rows(embeddings) if normalize else np.ascontiguousarray(embeddings, np.float32)
        )
        self._index = faiss.IndexFlatIP(self._embeddings.shape[1])
        self._index.add(self._embeddings)

    @classmethod
    def build(
        cls,
        product_ids: Sequence[str],
        texts: Sequence[str],
        encoder: Encoder,
        *,
        show_progress: bool = False,
    ) -> DenseRetriever:
        return cls(product_ids, encoder.encode(texts, show_progress=show_progress), encoder)

    def search_batch(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Search many queries at once. Blank queries give an empty list."""
        results: list[list[tuple[str, float]]] = [[] for _ in queries]
        live = [i for i, q in enumerate(queries) if q.strip()]
        if not live or k <= 0:
            return results
        vectors = _unit_rows(self.encoder.encode([queries[i] for i in live]))
        scores, indices = self._index.search(vectors, min(k, len(self.product_ids)))
        for row, i in enumerate(live):
            pairs = zip(indices[row], scores[row], strict=True)
            results[i] = [(self.product_ids[j], float(s)) for j, s in pairs if j >= 0]
        return results

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_batch([query], k)[0]

    def save(self, directory: str | Path, model_name: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMBEDDINGS_FILE, self._embeddings)
        (directory / IDS_FILE).write_text(json.dumps(self.product_ids), encoding="utf-8")
        meta = {
            "model": model_name,
            "n_products": len(self.product_ids),
            "dim": int(self._embeddings.shape[1]),
            "ids_fingerprint": ids_fingerprint(self.product_ids),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        encoder: Encoder,
        *,
        expected_model: str | None = None,
        expected_ids: Sequence[str] | None = None,
    ) -> DenseRetriever:
        """Load a saved index, refusing one built with another model or for another corpus."""
        directory = Path(directory)
        try:
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            product_ids = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
            embeddings = np.load(directory / EMBEDDINGS_FILE)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"no dense index in {directory}; build it first") from error
        if expected_model is not None and meta["model"] != expected_model:
            raise ValueError(f"index was built with {meta['model']}, config wants {expected_model}")
        if expected_ids is not None and meta["ids_fingerprint"] != ids_fingerprint(expected_ids):
            raise ValueError("index was built for a different product set; rebuild it")
        return cls(product_ids, embeddings, encoder, normalize=False)  # saved rows are unit length
