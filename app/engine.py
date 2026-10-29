"""Search engines behind the UI: the real pipeline when data exists, a sample-catalogue demo otherwise.

Both return the same JSON-ready dict, so the front end does not care which one answered.
"""

from __future__ import annotations

import json
import os
import re
import time

import pandas as pd

from retailmind.config import REPO_ROOT, load_ablations, load_config
from retailmind.kg.filter import ConstraintFilter
from retailmind.kg.graph import ProductKG
from retailmind.query.lexicon import BASE_COLORS, Lexicon, tokens
from retailmind.query.parser import QueryParser
from retailmind.retrieval.bm25 import BM25Retriever
from retailmind.retrieval.fusion import rrf

from demo_catalog import CATALOG, EXAMPLES  # app/ is on sys.path (see server.py)

KIND_LABELS = {
    "price": "price",
    "delivery": "delivery",
    "in_stock": "stock",
    "brand": "brand",
    "color": "colour",
    "material": "material",
}

# Headline numbers from REPORT.md for runs whose result folders are not committed.
REPORT_FALLBACK = {
    "A3 · no filter": {"ndcg@10": 0.3457, "mrr@10": 0.3892, "recall@100": 0.8006,
                       "violation@10": 0.57, "clean@10": 0.02, "source": "report"},
    "A6 · full system": {"ndcg@10": 0.6479, "mrr@10": 0.7305, "recall@100": 0.8347,
                         "violation@10": 0.03, "clean@10": 0.95, "source": "report"},
}  # fmt: skip


def constraint_chips(constraints) -> list[dict]:
    """Flatten a Constraints object into labelled chips for the UI."""
    chips: list[dict] = []
    for kind in ("brand", "color", "material", "size"):
        for value in getattr(constraints, kind):
            chips.append({"kind": kind, "label": kind.replace("color", "colour"), "value": value})
    lo, hi = constraints.price_min, constraints.price_max
    if lo is not None and hi is not None:
        chips.append({"kind": "price", "label": "price", "value": f"${lo:g} – ${hi:g}"})
    elif hi is not None:
        chips.append({"kind": "price", "label": "price", "value": f"≤ ${hi:g}"})
    elif lo is not None:
        chips.append({"kind": "price", "label": "price", "value": f"≥ ${lo:g}"})
    if constraints.delivery_max_days is not None:
        days = constraints.delivery_max_days
        chips.append({"kind": "delivery", "label": "delivery", "value": f"≤ {days} days"})
    if constraints.in_stock_only:
        chips.append({"kind": "in_stock", "label": "stock", "value": "in stock"})
    for term in constraints.exclude_terms:
        chips.append({"kind": "exclude", "label": "exclude", "value": term})
    return chips


def _overlap(query: str, text: str) -> float:
    q, t = set(tokens(query)), set(tokens(text))
    return len(q & t) / len(q) if q else 0.0


def checks_for(constraints, p: dict, violations: list[str]) -> list[dict]:
    """Requested-vs-actual rows for the product drawer."""
    if constraints is None:
        return []
    bad = set(violations)
    rows = []
    lo, hi = constraints.price_min, constraints.price_max
    if lo is not None or hi is not None:
        want = f"${lo:g} – ${hi:g}" if lo is not None and hi is not None else (
            f"≤ ${hi:g}" if hi is not None else f"≥ ${lo:g}")
        rows.append({"kind": "Price", "want": want, "have": f"${p['price']:.2f}", "ok": "price" not in bad})
    if constraints.delivery_max_days is not None:
        rows.append({"kind": "Delivery", "want": f"≤ {constraints.delivery_max_days} days",
                     "have": f"{p['delivery_days']} days", "ok": "delivery" not in bad})
    if constraints.in_stock_only:
        rows.append({"kind": "Stock", "want": "in stock",
                     "have": "in stock" if p["in_stock"] else "out of stock", "ok": "in_stock" not in bad})
    for kind, label, have in (("brand", "Brand", p.get("brand")), ("color", "Colour", p.get("color")),
                              ("material", "Material", p.get("material"))):
        wanted = getattr(constraints, kind)
        if wanted:
            rows.append({"kind": label, "want": ", ".join(wanted), "have": have or "unknown", "ok": kind not in bad})
    return rows


def _hit(p: dict, rank: int, score: float, violations: list[str], constraints=None) -> dict:
    return {
        "checks": checks_for(constraints, p, violations),
        "rank": rank,
        "product_id": p["product_id"],
        "title": p["title"],
        "brand": p.get("brand"),
        "color": p.get("color"),
        "material": p.get("material"),
        "category": p.get("category"),
        "price": p["price"],
        "delivery_days": p["delivery_days"],
        "in_stock": p["in_stock"],
        "score": round(float(score), 3),
        "violations": [KIND_LABELS.get(v, v) for v in violations],
        "passed": not violations,
    }


class DemoEngine:
    """Runs the project's real parser, BM25, fusion, knowledge graph and filter on a sample catalogue."""

    mode = "demo"
    llm_ok = False

    def __init__(self) -> None:
        self.runs = load_ablations()
        self.catalog = {p["product_id"]: p for p in CATALOG}
        ids = list(self.catalog)
        texts = [
            f"{p['title']} {p['brand']} {p['color']} {p['material']} {p['category']}"
            for p in CATALOG
        ]
        self.texts = dict(zip(ids, texts, strict=True))
        self.bm25 = BM25Retriever(ids, texts)
        lexicon = Lexicon(
            brands=frozenset(" ".join(tokens(p["brand"])) for p in CATALOG),
            colors=frozenset(" ".join(tokens(p["color"])) for p in CATALOG) & BASE_COLORS,
        )
        self.parser = QueryParser(lexicon)
        products = pd.DataFrame(
            {
                "product_id": ids,
                "brand": [p["brand"] for p in CATALOG],
                "color": [p["color"] for p in CATALOG],
                "text": texts,
            }
        )
        attributes = pd.DataFrame(
            {
                "product_id": ids,
                "price": [p["price"] for p in CATALOG],
                "delivery_days": [p["delivery_days"] for p in CATALOG],
                "in_stock": [p["in_stock"] for p in CATALOG],
            }
        )
        self.kg = ProductKG.from_products(products, attributes, lexicon)
        self.filter = ConstraintFilter(
            self.kg,
            mode="lenient",
            penalty=0.5,
            hard_penalty=100.0,
            enabled=["price", "delivery", "in_stock", "brand", "color", "material"],
        )

    @staticmethod
    def _decompose(query: str) -> list[str]:
        """Heuristic stand-in for the LLM decomposer: split 'a X and a Y' into separate needs."""
        parts = re.split(r"\s+(?:and|plus|&)\s+(?=(?:a|an|some|the)\s)", query.strip(), flags=re.I)
        parts = [p.strip(" ,") for p in parts if len(p.split()) >= 2]
        return parts if len(parts) > 1 else [query]

    def _retrieve(self, subs: list[str]) -> list[tuple[str, float]]:
        lists = [r for r in (self.bm25.search(s, 40) for s in subs if s.strip()) if r]
        if not lists:
            return []
        return lists[0] if len(lists) == 1 else rrf(lists, k=60, top_k=40)

    def search(self, query: str, run: str = "A6", top: int = 10) -> dict:
        t0 = time.perf_counter()
        comps = set(self.runs[run].components) if run in self.runs else set()
        parsed = self.parser.parse(query, "demo")
        search_text = parsed.product_terms if "parse" in comps else query
        subs = self._decompose(search_text) if "decompose" in comps else [search_text]
        trace: list[str] = []
        if "parse" in comps:
            trace.append(f"parse → {len(parsed.constraints.active())} constraint(s) found")
        if len(subs) > 1:
            trace.append(f"decompose → {len(subs)} sub-queries: " + " | ".join(subs))

        def rank(texts: list[str], use_constraints: bool):
            ranked = self._retrieve(texts)
            if "rerank" in comps and ranked:  # lexical proxy for the cross-encoder
                top_score = ranked[0][1] or 1.0
                ranked = sorted(
                    (
                        (p, 6 * _overlap(texts[0], self.texts[p]) + 4 * s / top_score)
                        for p, s in ranked
                    ),
                    key=lambda x: -x[1],
                )
            viol: dict[str, list[str]] = {}
            if "filter" in comps and use_constraints:
                ranked, viol = self.filter.apply(parsed.constraints, ranked)
            elif "filter" in comps:  # relaxed retry: keep the order, still flag what fails
                enabled = self.filter.enabled
                for pid, _ in ranked:
                    found = self.kg.violations(pid, parsed.constraints, enabled)
                    if found:
                        viol[pid] = found
            return ranked, viol

        ranked, viol = rank(subs, True)
        iterations = 1
        if "agent" in comps:
            for attempt in range(1, 3):
                good = sum(1 for p, _ in ranked[:10] if p not in viol)
                trace.append(f"self-check {attempt}: {good} compliant result(s), need 3")
                if good >= 3:
                    break
                trace.append("too few → relax to the raw query and retry")
                ranked, viol = rank([query], False)
                iterations += 1

        hits = [
            _hit(self.catalog[pid], n, score, viol.get(pid, []), parsed.constraints)
            for n, (pid, score) in enumerate(ranked[:top], 1)
        ]
        answer = self._answer(hits) if "generate" in comps else None
        return {
            "mode": self.mode,
            "run": run,
            "components": sorted(comps),
            "query": query,
            "search_text": search_text,
            "sub_queries": subs if len(subs) > 1 else [],
            "constraints": constraint_chips(parsed.constraints),
            "soft_preferences": parsed.soft_preferences,
            "hits": hits,
            "answer": answer,
            "answer_source": "template (demo mode, no LLM)" if answer else None,
            "citation_ids": [h["product_id"] for h in hits if h["passed"]][:3],
            "iterations": iterations,
            "trace": trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "funnel": {"candidates": len(ranked), "violating": sum(1 for p, _ in ranked if p in viol)},
        }

    @staticmethod
    def _answer(hits: list[dict]) -> str:
        ok = [h for h in hits if h["passed"]][:3]
        if not ok:
            return (
                "None of the retrieved products clearly satisfy every requirement, so I won't "
                "recommend one. Try relaxing the price, delivery or colour."
            )
        lines = [
            f"{h['title']} (${h['price']:.2f}, ships in {h['delivery_days']} day(s), "
            f"{'in stock' if h['in_stock'] else 'out of stock'}) [{n}]"
            for n, h in enumerate(ok, 1)
        ]
        # citation numbers refer to the position among the compliant shown products
        text = "Best match: " + lines[0] + "."
        if len(lines) > 1:
            text += " Also consider: " + "; ".join(lines[1:]) + "."
        return text


class LiveEngine(DemoEngine):
    """The real RetailMind pipeline over the ESCI corpus (needs the data and indexes built)."""

    mode = "live"

    def __init__(self) -> None:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        self.cfg = load_config()
        self.runs = load_ablations()
        processed = self.cfg.resolve(self.cfg.paths.processed_dir)
        self.meta = pd.read_parquet(processed / "products.parquet").set_index("product_id")
        self.attrs = pd.read_parquet(processed / "attributes.parquet").set_index("product_id")
        self.pipelines: dict[str, object] = {}
        self.llm_ok = bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"))

    def _pipeline(self, run: str):
        if run not in self.pipelines:
            from retailmind.pipeline import Pipeline

            comps = set(self.runs[run].components)
            if not self.llm_ok:
                comps -= {"decompose", "generate", "agent"}
            pipe = Pipeline.from_config(self.cfg, comps)
            for pid, title in self.meta["title"].items():  # real titles for the LLM to cite
                pipe.products[pid]["title"] = title
            self.pipelines[run] = pipe
        return self.pipelines[run]

    def search(self, query: str, run: str = "A6", top: int = 10) -> dict:
        t0 = time.perf_counter()
        pipe = self._pipeline(run)
        resp = pipe.search(query, "ui")
        constraints = resp.parsed.constraints if resp.parsed else None
        hits = []
        for hit in resp.hits[:top]:
            pid = hit.product_id
            m, a = self.meta.loc[pid], self.attrs.loc[pid]
            viol: list[str] = []
            if pipe.filter is not None and constraints is not None and pipe.filter.kg.has(pid):
                viol = pipe.filter.kg.violations(pid, constraints, pipe.filter.enabled)
            product = {
                "product_id": pid,
                "title": m["title"],
                "brand": m["brand"],
                "color": m["color"],
                "price": float(a["price"]),
                "delivery_days": int(a["delivery_days"]),
                "in_stock": bool(a["in_stock"]),
            }
            hits.append(_hit(product, hit.rank, hit.scores.rerank or 0.0, viol, constraints))
        return {
            "mode": self.mode,
            "run": run,
            "components": sorted(pipe.components),
            "query": query,
            "search_text": resp.parsed.product_terms if resp.parsed else query,
            "sub_queries": [],
            "constraints": constraint_chips(constraints) if constraints else [],
            "soft_preferences": resp.parsed.soft_preferences if resp.parsed else [],
            "hits": hits,
            "answer": resp.answer or None,
            "answer_source": os.getenv("LLM_MODEL") if resp.answer else None,
            "citation_ids": [h["product_id"] for h in hits][: self.cfg.llm.max_context_products],
            "iterations": resp.iterations,
            "trace": resp.trace,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "funnel": {
                "candidates": len(resp.hits),
                "violating": sum(1 for h in hits if h["violations"]),
            },
        }


def load_engine():
    """Live engine if the processed data is present, else the demo engine."""
    processed = REPO_ROOT / "data" / "processed"
    ready = all(
        (processed / f).exists() for f in ("products.parquet", "attributes.parquet", "lexicon.json")
    )
    if os.getenv("RETAILMIND_MODE", "").lower() != "demo" and ready:
        try:
            return LiveEngine()
        except Exception as error:  # noqa: BLE001 - fall back rather than show a blank UI
            print(f"[retailmind] live engine failed ({error}); using demo mode")
    return DemoEngine()


def load_results() -> dict:
    """Summarise committed evaluation runs for the Results tab."""
    found: dict[str, dict] = {}
    for folder in sorted((REPO_ROOT / "results").glob("*")):
        if (folder / "metrics.json").exists():
            found[folder.name] = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    descriptions = {n: r.description for n, r in load_ablations().items()}
    ablation = [
        {"run": n, "description": descriptions[n], **found[f"{n}_test"]["metrics"]}
        for n in ("A0", "A1", "A2", "A3", "A4")
        if f"{n}_test" in found
    ]
    sc = []
    for key, label in (
        ("A3_sc_test", "A3 · no filter"),
        ("A4_sc_test", "A4 · + filter"),
        ("A6_sc_test", "A6 · full system"),
    ):
        if key in found:
            d = found[key]
            sc.append({"run": label, **d["metrics"], **d.get("constraint_metrics", {})})
        elif label in REPORT_FALLBACK:
            sc.append({"run": label, **REPORT_FALLBACK[label]})
    return {"ablation": ablation, "sc100": sc}
