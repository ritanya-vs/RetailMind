"""Try filter settings on a saved A4 run, without running retrieval or re-ranking again.

run_retrieval.py --run A4 saves the ranking before filtering (run_prefilter.json). This script
re-applies the filter with many settings and prints one table. Tune on dev only.

Usage:
    python scripts/sweep_filter.py --split dev
    python scripts/sweep_filter.py --dataset sc --split dev
"""

import argparse
import json
import statistics
import sys

import pandas as pd

from retailmind.config import load_config
from retailmind.eval.metrics import evaluate_results, load_split
from retailmind.eval.synthetic import load_sc, violation_rates
from retailmind.kg.build import load_kg, load_lexicon, parse_queries
from retailmind.kg.filter import ConstraintFilter

# Re-ranker scores are raw logits (roughly -11 to +10), so useful penalties reach into the tens.
PENALTIES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
SOFT_PENALTIES = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)  # used together with hard_penalty=100
HARD_PENALTY = 100.0
KIND_SETS = {
    "esci": {
        "all": ["price", "delivery", "in_stock", "brand", "color", "material"],
        "brand": ["brand"],
        "color": ["color"],
        "material": ["material"],
        "color+material": ["color", "material"],
    },
    "sc": {
        "price+delivery+stock": ["price", "delivery", "in_stock"],
        "all": ["price", "delivery", "in_stock", "brand", "color", "material"],
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="A4")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--dataset", choices=["esci", "sc"], default="esci")
    args = parser.parse_args(argv)

    cfg = load_config()
    processed = cfg.resolve(cfg.paths.processed_dir)
    name = f"{args.run}_{args.split}" if args.dataset == "esci" else f"{args.run}_sc_{args.split}"
    out_dir = cfg.resolve(cfg.paths.results_dir) / name
    try:
        saved = json.loads((out_dir / "run_prefilter.json").read_text(encoding="utf-8"))
        truth = {}
        if args.dataset == "esci":
            queries, qrels = load_split(processed, args.split)
        else:
            queries, qrels, truth = load_sc(processed, args.split)
        lexicon = load_lexicon(cfg)
        kg = load_kg(cfg, lexicon)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun: python scripts/run_retrieval.py --run {args.run} "
              f"--dataset {args.dataset} --split {args.split}", file=sys.stderr)  # fmt: skip
        return 1

    prefilter = {q: list(hits.items()) for q, hits in saved.items()}
    parsed = parse_queries(lexicon, queries["query_id"].tolist(), queries["query"].tolist())
    # the parser must see the original text; saved rankings are keyed by query id
    top_scores = [s for hits in prefilter.values() for _, s in hits[:10]]
    spread = f"median {statistics.median(top_scores):.3f}, "
    spread += f"min {min(top_scores):.3f}, max {max(top_scores):.3f}"
    print(f"Top-10 re-ranker scores: {spread} (penalties are in these units)")

    def score(results):
        means, _ = evaluate_results(results, qrels)
        row = {
            "ndcg@10": means["ndcg@10"],
            "mrr@10": means["mrr@10"],
            "recall@100": means["recall@100"],
        }
        row.update(violation_rates(results, truth, kg) if truth else {})
        return row

    rows = [
        {"kinds": "(none)", "mode": "no filter", "soft": None, "hard": None, **score(prefilter)}
    ]

    def run(label, kinds, mode, soft, hard):
        flt = ConstraintFilter(kg, mode=mode, penalty=soft, hard_penalty=hard, enabled=kinds)
        results = {q: flt.apply(parsed[q].constraints, hits)[0] for q, hits in prefilter.items()}
        rows.append({"kinds": label, "mode": mode, "soft": soft, "hard": hard, **score(results)})

    for label, kinds in KIND_SETS[args.dataset].items():
        run(label, kinds, "strict", 0.0, None)
        for penalty in PENALTIES:  # one penalty for every constraint kind
            run(label, kinds, "lenient", penalty, None)
        if args.dataset == "sc" and label == "all":  # hard limits sink, brand/color/material nudge
            for soft in SOFT_PENALTIES:
                run(label, kinds, "lenient", soft, HARD_PENALTY)

    table = pd.DataFrame(rows)
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 140
    print(table.to_string(index=False))
    best = table.iloc[1:].sort_values("ndcg@10", ascending=False).iloc[0]
    base = table.iloc[0]["ndcg@10"]
    gain = best["ndcg@10"] - base
    setting = f"{best['kinds']} / {best['mode']} / soft {best['soft']} / hard {best['hard']}"
    print(f"\nBest by nDCG@10: {setting} -> {best['ndcg@10']:.4f} ({gain:+.4f} vs no filter)")
    (out_dir / "filter_sweep.json").write_text(
        table.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(f"Saved {out_dir / 'filter_sweep.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
