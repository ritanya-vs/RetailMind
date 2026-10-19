"""The full RetailMind pipeline, assembled from the stages, for the demo and A5/A6 evaluation.

This wires retrieval -> fusion -> re-rank -> parse -> filter -> (decompose, agent, generate) into
one object. Which stages run is controlled by the `components` set, so the same class serves every
ablation. The heavy models (dense index, cross-encoder, LLM) are loaded lazily and only if their
stage is switched on, so a BM25-only run does not load torch or need an LLM key.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from retailmind.agent.loop import AgentPolicy, SearchAgent
from retailmind.config import Config
from retailmind.generate.answer import AnswerGenerator
from retailmind.generate.llm import LLMClient, make_client
from retailmind.kg.filter import ConstraintFilter
from retailmind.kg.graph import ProductKG
from retailmind.query.decompose import QueryDecomposer
from retailmind.query.lexicon import Lexicon
from retailmind.query.parser import QueryParser
from retailmind.rerank.cross_encoder import CrossEncoderScorer, Reranker
from retailmind.retrieval.bm25 import BM25Retriever
from retailmind.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from retailmind.retrieval.fusion import rrf
from retailmind.schemas import (
    PipelineResponse,
    SearchHit,
    StageScores,
)

Ranked = list[tuple[str, float]]


class Pipeline:
    """Build with Pipeline.from_config(...). Call .search(query) for one PipelineResponse."""

    def __init__(
        self,
        cfg: Config,
        components: set[str],
        products: pd.DataFrame,
        *,
        lexicon: Lexicon | None = None,
        kg: ProductKG | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.cfg = cfg
        self.components = components
        self.ids = products["product_id"].tolist()
        self.texts = dict(zip(self.ids, products["text"], strict=True))
        self.depth = max(cfg.retrieval.bm25_k, cfg.retrieval.dense_k)

        self.bm25 = (
            BM25Retriever(self.ids, products["text"].tolist()) if "bm25" in components else None
        )
        self.dense = None
        if "dense" in components:
            encoder = SentenceTransformerEncoder(cfg.retrieval.dense_model)
            self.dense = DenseRetriever.load(
                cfg.resolve(cfg.paths.index_dir) / "dense",
                encoder,
                expected_model=cfg.retrieval.dense_model,
                expected_ids=self.ids,
            )
        self.reranker = (
            Reranker(CrossEncoderScorer(cfg.rerank.model), self.texts)
            if "rerank" in components
            else None
        )
        self.parser = QueryParser(lexicon) if lexicon and "parse" in components else None
        self.filter = None
        if "filter" in components and kg is not None:
            self.filter = ConstraintFilter(
                kg,
                mode=cfg.filter.mode,
                penalty=cfg.filter.penalty,
                hard_penalty=cfg.filter.hard_penalty,
                enabled=cfg.filter.constraints,
            )
        self.decomposer = QueryDecomposer(llm) if llm and "decompose" in components else None
        self.generator = (
            AnswerGenerator(llm, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)
            if llm and "generate" in components
            else None
        )
        self.products = {pid: {"title": "", "text": self.texts[pid]} for pid in self.ids}

    def _retrieve_one(self, text: str) -> Ranked:
        lists: list[Ranked] = []
        if self.bm25 is not None:
            lists.append(self.bm25.search(text, self.cfg.retrieval.bm25_k))
        if self.dense is not None:
            lists.append(self.dense.search(text, self.cfg.retrieval.dense_k))
        if not lists:
            return []
        if len(lists) == 1:
            return lists[0][: self.depth]
        return rrf(lists, k=self.cfg.retrieval.rrf_k, top_k=self.depth)

    def _retrieve(self, sub_queries: Sequence[str]) -> Ranked:
        """Retrieve each sub-query and fuse the lists, so a multi-need query covers all needs."""
        per_sub = [self._retrieve_one(text) for text in sub_queries if text.strip()]
        if len(per_sub) <= 1:
            return per_sub[0] if per_sub else []
        return rrf(per_sub, k=self.cfg.retrieval.rrf_k, top_k=self.depth)

    def search(self, query: str, query_id: str = "") -> PipelineResponse:
        parsed = self.parser.parse(query, query_id) if self.parser else None
        search_text = parsed.product_terms if parsed else query
        sub_queries = [search_text]
        if self.decomposer is not None:
            sub_queries = self.decomposer.decompose(query, fallback=search_text)

        def retrieve(_relaxed: str, iteration: int) -> Ranked:
            subs = [_relaxed] if iteration > 0 else sub_queries
            ranked = self._retrieve(subs)
            if self.reranker is not None and ranked:
                ranked = self.reranker.rerank_batch([subs[0]], [ranked], self.cfg.rerank.top_k)[0]
            if self.filter is not None and parsed is not None:
                ranked = self.filter.apply(parsed.constraints, ranked)[0]
            return ranked

        trace: list[str] = []
        if "agent" in self.components:
            policy = AgentPolicy(
                min_good_results=self.cfg.agent.min_good_results,
                min_top_score=self.cfg.agent.min_rerank_score,
                max_iterations=self.cfg.agent.max_iterations,
            )
            agent = SearchAgent(
                retrieve,
                lambda q, ranked: self._answer(q, ranked, parsed),
                policy,
                relaxed_query=lambda _q: query,  # relax = drop parsed terms, use the raw query
            )
            outcome = agent.run(query)
            ranked, answer, iterations, trace = (
                outcome.results,
                outcome.answer,
                outcome.iterations,
                outcome.trace,
            )
        else:
            ranked = retrieve(search_text, 0)
            answer = self._answer(query, ranked, parsed) if self.generator else None
            iterations = 1

        return self._response(query_id, parsed, ranked, answer, iterations, trace)

    def _answer(self, query: str, ranked: Ranked, parsed) -> str:
        if self.generator is None:
            return ""
        constraints = parsed.constraints if parsed else None
        answer, _ = self.generator.generate(
            query,
            [pid for pid, _ in ranked],
            self.products,
            constraints=constraints,
            max_products=self.cfg.llm.max_context_products,
        )
        return answer

    def _response(self, query_id, parsed, ranked, answer, iterations, trace) -> PipelineResponse:
        hits = [
            SearchHit(product_id=pid, rank=rank, scores=StageScores(rerank=score))
            for rank, (pid, score) in enumerate(ranked, start=1)
        ]
        return PipelineResponse(
            query_id=query_id,
            parsed=parsed,
            hits=hits,
            answer=answer,
            iterations=iterations,
            trace=trace,
        )

    @classmethod
    def from_config(
        cls, cfg: Config, components: set[str], *, require_llm: bool | None = None
    ) -> Pipeline:
        """Load products, and only the lexicon/KG/LLM that the chosen components need."""
        processed = cfg.resolve(cfg.paths.processed_dir)
        products = pd.read_parquet(processed / "products.parquet")
        lexicon = kg = llm = None
        if components & {"parse", "filter"}:
            lexicon = Lexicon.load(processed / "lexicon.json")
        if "filter" in components:
            attributes = pd.read_parquet(processed / "attributes.parquet")
            kg = ProductKG.from_products(products, attributes, lexicon)
        needs_llm = bool(components & {"decompose", "generate"})
        if needs_llm if require_llm is None else require_llm:
            llm = make_client(cfg.llm, cache_dir=cfg.resolve(cfg.llm.cache_dir))
        return cls(cfg, components, products, lexicon=lexicon, kg=kg, llm=llm)
