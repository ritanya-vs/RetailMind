"""BM25 keyword retrieval over the product text."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from retailmind.retrieval.tokenize import tokenize


class _LuceneBM25(BM25Okapi):
    """BM25Okapi with the Lucene/Elasticsearch idf, which is never negative.

    The stock rank_bm25 idf is log(N - n + 0.5) - log(n + 0.5), which is zero or negative for any
    word found in half the documents or more and then gets patched with an epsilon rule.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        idf_sum = 0.0
        for word, freq in nd.items():
            idf = float(np.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5)))
            self.idf[word] = idf
            idf_sum += idf
        self.average_idf = idf_sum / len(nd) if nd else 0.0


class BM25Retriever:
    """Okapi BM25 over a fixed corpus. Ties are broken by corpus order, so results are stable."""

    def __init__(
        self, product_ids: Sequence[str], texts: Sequence[str], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        if len(product_ids) != len(texts):
            raise ValueError("product_ids and texts must have the same length")
        if not product_ids:
            raise ValueError("cannot build an index over an empty corpus")
        self.product_ids = list(product_ids)
        self._bm25 = _LuceneBM25([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to k (product_id, score) pairs, best first, leaving out zero scores."""
        tokens = tokenize(query)
        if not tokens or k <= 0:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens))
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]  # score descending, then corpus order
        return [(self.product_ids[i], float(scores[i])) for i in top if scores[i] > 0]
