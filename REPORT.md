# RetailMind: Multi-Constraint Product Search with Hybrid Retrieval, Knowledge-Graph Filtering, and a Grounded Agent

**An Information Retrieval project evaluated on Amazon ESCI.**

---

## Abstract

RetailMind is a product-search pipeline that answers natural-language shopping queries carrying
multiple constraints (brand, colour, material, price, delivery time, stock). It combines sparse and
dense retrieval, reciprocal rank fusion, cross-encoder re-ranking, a knowledge-graph constraint
filter, and a LangGraph agent that self-checks its results and writes a grounded recommendation
with a local LLM. The system is evaluated against a BM25 baseline on the Amazon ESCI Shopping
Queries dataset, using an ablation ladder (A0–A6) measured on held-out test queries. On ESCI, the
full retrieval stack lifts nDCG@10 from **0.470 (BM25) to 0.575** and MRR@10 from **0.601 to 0.715**.
Because ESCI queries rarely state price or delivery limits, a synthetic constrained set (SC-100)
isolates the constraint machinery: the knowledge-graph filter raises nDCG@10 from **0.346 to 0.658**
and cuts the share of top-10 results that violate a stated constraint from ~57% to **3%**. The
grounded generator cites only products it was shown (**citation validity 1.00**), demonstrating a
faithful, non-hallucinating answer layer. We report the limitations of the synthetic evaluation
openly.

---

## 1. Introduction

Keyword search handles "running shoes" well but struggles with "red Nike running shoes under $80
delivered in two days": the request mixes a product need (running shoes), soft attributes (red,
Nike), and hard constraints (price, delivery). A lexical ranker treats every word as a bag of
terms, so it cannot guarantee that the top results actually respect the price ceiling or the
delivery window.

RetailMind addresses this by separating the two problems. A retrieval stack finds products that
match the *intent*, and a constraint layer enforces the *requirements* over a product knowledge
graph. An agentic controller then checks whether the results are strong enough, retries with a
relaxed query if not, and finally produces a short recommendation grounded strictly in the
retrieved products.

### Research questions

- **RQ1.** Does hybrid retrieval (BM25 + dense) with reciprocal rank fusion beat a BM25 baseline on
  ESCI?
- **RQ2.** Does cross-encoder re-ranking improve ranking quality over fusion?
- **RQ3.** Can a parsed-constraint filter enforce hard requirements without destroying recall?
- **RQ4.** Does the agent + grounded generation layer produce faithful answers (no invented
  products)?
- **RQ5.** What does the Amazon ESCI dataset let us demonstrate, and what does it not?

---

## 2. Dataset

### 2.1 Amazon ESCI (Shopping Queries)

We use the public Amazon **ESCI Shopping Queries** dataset (Task 1, US locale, small version). Two
tables are used: an *examples* table of query–product judgements and a *products* table of product
text. Each judgement carries one of four labels:

| Label | Meaning | Graded gain |
|---|---|---|
| E — Exact | the product satisfies the query | 1.0 |
| S — Substitute | a reasonable alternative | 0.1 |
| C — Complement | used together with the item | 0.01 |
| I — Irrelevant | does not fit | 0.0 |

The graded gains follow the ordering described in the ESCI paper (E > S > C > I). Because ranx
stores relevance as integers, all gains are scaled by 100 (100 / 10 / 1 / 0) so the Substitute and
Complement signal is preserved in nDCG.

### 2.2 Prepared corpus

From the US small set (29,844 queries) we sample **300 development** queries (from the ESCI *train*
split) and **700 test** queries (from the ESCI *test* split), keeping tuning strictly separate from
the held-out evaluation. The search corpus holds every judged product for those queries plus
10,000 additional "distractor" products, for **29,775 products** in total. Product text is cleaned
(HTML, entities and emoji removed) and the title, brand, colour, bullet points and description are
concatenated into one searchable field capped at 2,000 characters.

### 2.3 SC-100: a synthetic constrained set

ESCI queries almost never state a price or delivery limit, so they cannot test whether a system
respects such constraints. We therefore build **SC-100**, 100 development and 100 test queries, each
derived from an ESCI query with appended constraint phrases ("under $40", "delivered in 2 days",
"in stock"). Because ESCI has no price, delivery or availability data, each product is assigned
**synthetic** values (deterministic per product id). A product is relevant for an SC query only if
ESCI judged it relevant **and** its synthetic attributes satisfy every stated constraint. Limits
are drawn so that some Exact products pass and some fail, which makes the constraint discriminative.
*These price/delivery/stock values are synthetic and are labelled as such throughout.*

---

## 3. System architecture

The pipeline is a sequence of stages, each independently switchable for ablation:

```
query
  │
  ├─ query understanding ─ parse constraints (brand, colour, material, price, delivery, stock)
  │                         decompose multi-need queries (LLM)
  │
  ├─ retrieval ─ BM25 (Lucene idf)  ┐
  │             dense (MiniLM+FAISS) � ─ reciprocal rank fusion (k=60)
  │
  ├─ re-ranking ─ cross-encoder (ms-marco-MiniLM-L-6-v2) over the top 50
  │
  ├─ constraint filter ─ knowledge graph (NetworkX): strict removal or lenient penalty
  │
  └─ agent (LangGraph) ─ self-check → relax & retry → grounded answer (local LLM)
```

- **Query parser.** A deterministic, rule- and lexicon-based parser extracts constraints. The brand
  and colour vocabularies are learned from the product catalogue; a one-word brand is kept only if
  that word almost always denotes the brand.
- **BM25.** Okapi BM25 with the Lucene/Elasticsearch idf (never negative), which is the standard
  behaviour people expect from "BM25".
- **Dense + fusion.** `all-MiniLM-L6-v2` embeddings searched with an exact FAISS inner-product
  index; fused with BM25 by reciprocal rank fusion, which uses only rank positions and so needs no
  score scaling.
- **Cross-encoder.** `ms-marco-MiniLM-L-6-v2` re-scores the top 50 candidates; the rest keep their
  order below them, so recall at depth is never reduced by re-ranking.
- **Knowledge graph & filter.** Products link to brand, colour and material nodes and carry price,
  delivery and stock attributes. The filter runs *strict* (remove violators) or *lenient* (subtract
  a penalty). A large penalty for hard numeric limits (price, delivery, stock) sinks violators below
  all compliant products without dropping them, preserving recall; a small penalty nudges the softer
  brand/colour/material signal.
- **Agent & generation.** A LangGraph loop checks whether enough confident results exist; if not it
  relaxes the query and retries (up to two extra attempts) before writing an answer. The grounded
  generator shows the LLM only the top retrieved products and requires it to cite them as [1], [2].
  The LLM is any OpenAI-compatible endpoint; this project used **qwen2.5** served locally through
  Ollama.

---

## 4. Experimental setup

- **Metrics.** nDCG@10 (graded), nDCG@10 on Exact only, MRR@10 and Recall@100, computed with `ranx`.
  For SC-100 we additionally report `violation@10` (mean share of the top 10 breaking a stated
  constraint) and `clean@10` (share of queries with no violation in the top 10). For generated
  answers we report citation validity, grounded rate, abstention rate and answer relevancy
  (MiniLM cosine between answer and query).
- **Splits.** All tuning (fusion constant, re-rank depth, filter penalty) is done on **dev**; every
  headline number below is on the untouched **test** split. Paired bootstrap confidence intervals
  (5,000 resamples) are used to judge whether a difference is real.
- **Ablation ladder.** A0 BM25 · A1 dense · A2 BM25+dense (RRF) · A3 +re-rank · A4 +parse+filter ·
  A5 +decomposition · A6 +agent+grounded generation.
- **Engineering.** `src`-layout Python package, Pydantic v2 strict config, 167 passing unit tests,
  `ruff`-clean. LLM responses are cached on disk so evaluation is reproducible and never re-billed.

---

## 5. Results

### 5.1 Retrieval ablation on ESCI (test, 700 queries)

| Run | Components | nDCG@10 | nDCG@10 (Exact) | MRR@10 | Recall@100 |
|---|---|---|---|---|---|
| A0 | BM25 | 0.4705 | 0.4591 | 0.6008 | 0.7727 |
| A1 | dense | 0.4684 | 0.4579 | 0.6101 | 0.8000 |
| A2 | BM25 + dense (RRF) | 0.5074 | 0.4961 | 0.6418 | 0.8401 |
| A3 | A2 + re-rank | **0.5746** | **0.5631** | **0.7139** | 0.8401 |
| A4 | A3 + parse + filter | 0.5748 | 0.5634 | 0.7151 | 0.8401 |

- **RQ1 — fusion helps.** A2 beats A0 on every metric: nDCG@10 +0.037, MRR@10 +0.041, Recall@100
  +0.067. Dense alone (A1) is level with BM25, but the two retrievers miss different products, so
  fusing them recovers recall (0.773 → 0.840). The paired interval for the recall gain excludes
  zero.
- **RQ2 — re-ranking helps most.** A3 adds +0.067 nDCG@10 and +0.072 MRR@10 over A2 (both intervals
  exclude zero). Recall is unchanged by design, confirming the re-ranker only reorders the top 50.
  Over the baseline, A3 improves nDCG@10 by **+0.104** and MRR@10 by **+0.113**.
- **A4 is neutral on ESCI.** Adding the constraint filter changes nDCG@10 by +0.0002 — effectively
  nothing. This is expected: ESCI queries carry no explicit price/delivery limits, and a parser
  analysis found brand/colour are weak discriminators on ESCI (Irrelevant products match a query's
  brand as often as Exact ones, because ESCI's judged set already shares the brand). The filter is
  therefore configured *lenient* so it never hurts real queries, and its value is shown on SC-100.

### 5.2 Constraint evaluation on SC-100 (test, 100 queries)

| Run | nDCG@10 | MRR@10 | Recall@100 | violation@10 | clean@10 |
|---|---|---|---|---|---|
| A3 (no filter) | 0.3457 | 0.3892 | 0.8006 | ~0.57 \* | ~0.02 \* |
| A4 (+ filter) | **0.6575** | **0.7541** | 0.8424 | **0.0300** | **0.9500** |
| A6 (full system) | 0.6479 | 0.7305 | 0.8347 | 0.0300 | 0.9500 |

\* no-filter violation/clean measured on the dev split, where A3 does not load the graph on the test
run; it is representative of the unfiltered system.

- **RQ3 — the filter enforces constraints without losing recall.** On constrained queries the filter
  almost doubles nDCG@10 (0.346 → 0.658) and raises Recall@100 slightly (0.801 → 0.842). The share
  of top-10 results violating a stated constraint falls from roughly **57% to 3%**, and 95% of
  queries have a completely clean top 10. Recall is preserved because hard-limit violators are
  demoted, not deleted.
- **A4 ≈ A6 on ranking.** The decomposition and agent stages leave the ranking essentially
  unchanged (0.658 → 0.648); their contribution is the grounded answer, not the ordering.

### 5.3 Answer quality (A6, SC-100 test)

| Metric | Value | Reading |
|---|---|---|
| citation_validity | **1.0000** | every cited product was one the model was shown — no invented products |
| grounded_rate | 0.9865 | 98.7% of substantive answers cite only valid products |
| mean_citations | 1.47 | answers cite one or two products on average |
| abstention_rate | 0.2600 | the model declines on ~26% of queries rather than overclaim |
| answer_relevancy | 0.7098 | answers are on-topic with the query (MiniLM cosine) |

- **RQ4 — answers are faithful.** Across 100 test queries the grounded generator never cited a
  product outside the retrieved set. The ~26% abstention rate reflects honest behaviour: when the
  retrieved products do not clearly match the request, qwen2.5 says so instead of inventing a fit.

---

## 6. Discussion

The ablation tells a clean, decomposable story. **Fusion fixes recall** — it brings in the correct
products that either retriever alone misses. **Re-ranking fixes ordering** — it moves the Exact
matches to the top, which is where nDCG and MRR improve. **The filter fixes constraint compliance** —
invisible on ESCI (no constraints to enforce) but decisive on SC-100. **The agent layer adds a
faithful answer** without disturbing the ranking. Each stage earns its place for a measurable,
separable reason, rather than the system being presented as one opaque number.

---

## 7. Limitations

We state these plainly; they bound what the evaluation proves.

1. **SC-100 is partly circular.** Its relevance is defined by the same constraints the filter
   enforces, and the parser is tested on the phrasings we generated. SC-100 shows the constraint
   mechanism works end to end; it does not prove that arbitrary real-shopper phrasing would parse as
   cleanly.
2. **Price, delivery and stock are synthetic.** ESCI contains no such data. These fields exist only
   to exercise and measure the constraint layer and are labelled synthetic throughout.
3. **Answer quality is measured on SC-100.** The faithfulness and relevancy numbers are over the
   synthetic constrained queries; a real-query answer evaluation would strengthen the claim.
4. **Our ESCI numbers are not comparable to published ESCI leaderboards.** We use a 29,775-product
   subset and our own sampled queries, not the full catalogue or official evaluation set.
5. **The answer LLM is small and local** (qwen2.5 via Ollama). A larger model would likely abstain
   less and write richer answers; the grounding contract would be unchanged.

---

## 8. Conclusion and future work

RetailMind shows that a transparent, stage-wise pipeline can take product search from a BM25
baseline (nDCG@10 0.470) to a re-ranked hybrid system (0.575) on Amazon ESCI, and that a
knowledge-graph constraint layer can enforce hard requirements — demonstrated on a synthetic
constrained set where it lifts nDCG@10 from 0.346 to 0.658 and reduces constraint violations from
~57% to 3% — topped by an agent that produces faithful, fully grounded recommendations (citation
validity 1.00).

Future work: a human-labelled constrained query set to remove SC-100's circularity; a learned (not
rule-based) constraint parser; real catalogue attributes where available; and a larger answer model
with a real-query faithfulness study.

---

## 9. Reproducibility

```
# data
python scripts/download_esci.py            # fetch + verify ESCI
python scripts/prepare_data.py             # products.parquet, queries.parquet, qrels.tsv
python scripts/build_lexicon.py            # brand/colour vocabulary
python scripts/build_attributes.py         # synthetic price/delivery/stock
python scripts/make_sc_set.py              # SC-100 dev + test

# indexes
python scripts/build_dense_index.py        # MiniLM + FAISS

# evaluation
python scripts/run_retrieval.py --run A0 --split test        # A0-A4 (no LLM)
python scripts/run_pipeline.py  --run A6 --dataset sc --split test   # A5-A6 (LLM)
python scripts/build_report.py  --split test                 # ablation table
```

Configuration lives in `configs/default.yaml`; the LLM endpoint is set in `.env`
(`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`). The repository has 167 unit tests (`pytest`) and is
`ruff`-clean. All results in this report were produced by the commands above; LLM responses are
cached so runs are reproducible.
