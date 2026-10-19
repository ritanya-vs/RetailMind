"""Check the query parser against the ESCI judgments, without any manual labels.

For every constraint type it reports how many queries have one, and how often the judged products
of those queries satisfy it, split by ESCI label. A constraint is a good hard filter when Exact
products satisfy it almost always and Irrelevant products rarely do. If Exact products often fail
it, filtering on it would throw away correct answers.

Usage:
    python scripts/analyze_parser.py --split dev
"""

import argparse
import json
import sys

import pandas as pd

from retailmind.config import load_config
from retailmind.eval.metrics import load_split
from retailmind.query.analysis import KINDS, LABELS, agreement, group_judgments, product_info
from retailmind.query.lexicon import Lexicon
from retailmind.query.parser import QueryParser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--examples", type=int, default=8, help="examples to print per type")
    args = parser.parse_args(argv)

    cfg = load_config()
    processed = cfg.resolve(cfg.paths.processed_dir)
    try:
        lexicon = Lexicon.load(processed / "lexicon.json")
        products = pd.read_parquet(processed / "products.parquet")
        queries, qrels = load_split(processed, args.split)
    except (OSError, ValueError) as error:
        print(f"Failed: {error}\nRun scripts/prepare_data.py and scripts/build_lexicon.py.")
        return 1

    info = product_info(products)
    qparser = QueryParser(lexicon)
    parsed = {
        qid: qparser.parse(text, qid)
        for qid, text in zip(queries["query_id"], queries["query"], strict=True)
    }
    judged = group_judgments(qrels)

    n = len(parsed)
    print(f"Parser analysis on {n} {args.split} queries\n")
    names = ("brand", "color", "size", "material", "price", "delivery", "in_stock", "exclude")
    coverage = {
        name: sum(name in p.constraints.active() for p in parsed.values()) for name in names
    }
    print("Queries with each constraint: " + ", ".join(f"{k} {v}" for k, v in coverage.items()))
    constrained = sum(not p.constraints.is_empty() for p in parsed.values())
    print(f"Queries with any constraint: {constrained}\n")

    report = {"split": args.split, "n_queries": n, "coverage": coverage, "agreement": {}}
    for kind in KINDS:
        strict = agreement(parsed, judged, info, kind)
        loose = agreement(parsed, judged, info, kind, use_text=True)
        has_text_mode = kind in ("brand", "color")
        meaning = "has the excluded word" if kind == "exclude_terms" else "satisfies it"
        head = (
            f"== {kind}: {len(strict['queries'])} queries; share of judged products that {meaning}"
        )
        print(head + (" (product field | field or title)" if has_text_mode else ""))
        for label in LABELS:
            matched, total = strict["counts"][label]
            shown = f"{matched / total:6.1%}" if total else "   n/a"
            extra = ""
            if has_text_mode and total:
                extra = f" | {loose['counts'][label][0] / total:6.1%}"
            print(f"   {label}: {shown}{extra}  (n={total})")
        for qid in strict["queries"][: args.examples]:
            print(f"   e.g. {parsed[qid].raw!r} -> {getattr(parsed[qid].constraints, kind)}")
        print()
        rates = {k: (round(v, 3) if v is not None else None) for k, v in strict["rates"].items()}
        loose_rates = {
            k: (round(v, 3) if v is not None else None) for k, v in loose["rates"].items()
        }
        report["agreement"][kind] = {
            "queries": len(strict["queries"]),
            "rates": rates,
            "rates_with_text": loose_rates,
        }

    out = cfg.resolve(cfg.paths.results_dir) / f"parser_analysis_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
