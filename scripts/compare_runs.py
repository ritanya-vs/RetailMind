"""Compare two scored runs query by query, with a 95% bootstrap interval for the difference.

Usage:
    python scripts/compare_runs.py A2 A3 --split dev      shows A3 minus A2
"""

import argparse
import json
import sys

from retailmind.config import load_config
from retailmind.eval.compare import paired_comparison
from retailmind.eval.metrics import METRIC_NAMES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", help="run to compare against, e.g. A2")
    parser.add_argument("other", help="run being judged, e.g. A3")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args(argv)

    cfg = load_config()
    results = cfg.resolve(cfg.paths.results_dir)
    try:
        base = json.loads((results / f"{args.base}_{args.split}" / "per_query.json").read_text())
        other = json.loads((results / f"{args.other}_{args.split}" / "per_query.json").read_text())
    except OSError as error:
        print(f"Failed: {error}", file=sys.stderr)
        return 1

    print(f"{args.other} minus {args.base} on {args.split} (95% bootstrap interval)")
    for metric in METRIC_NAMES:
        r = paired_comparison(base[metric], other[metric])
        verdict = "real" if r["ci_low"] > 0 or r["ci_high"] < 0 else "not clear"
        print(
            f"  {metric:14s} {r['mean_diff']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            f"  wins {r['wins']} losses {r['losses']}  {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
