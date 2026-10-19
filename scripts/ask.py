"""Ask the pipeline one question and print the parsed constraints, top hits, and the answer.

Loads .env for the LLM settings. Pick how much of the pipeline runs with --run (an ablation name).

Usage:
    python scripts/ask.py "red nike running shoes under $80 delivered in 2 days"
    python scripts/ask.py "a yoga mat and a water bottle" --run A6
"""

import argparse
import sys

from dotenv import load_dotenv

from retailmind.config import load_ablations, load_config
from retailmind.pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", help="the shopper's search")
    parser.add_argument("--run", default="A6", help="ablation name (default A6, the full system)")
    parser.add_argument("--top", type=int, default=5, help="how many hits to print")
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = load_config()
    runs = load_ablations()
    if args.run not in runs:
        print(f"Unknown run {args.run!r}; choose from {sorted(runs)}", file=sys.stderr)
        return 1
    components = set(runs[args.run].components)

    try:
        pipeline = Pipeline.from_config(cfg, components)
        response = pipeline.search(args.query, "demo")
    except Exception as error:  # noqa: BLE001 - a demo should print the reason, not a traceback
        print(f"Failed: {error}", file=sys.stderr)
        return 1

    if response.parsed is not None:
        active = response.parsed.constraints.active()
        print(f"Parsed constraints: {active or 'none'}")
        if active:
            print(f"  {response.parsed.constraints.model_dump(exclude_defaults=True)}")
        print(f"Search text: {response.parsed.product_terms!r}")
    print(f"\nTop {args.top} products:")
    for hit in response.hits[: args.top]:
        print(f"  {hit.rank}. {hit.product_id}  (score {hit.scores.rerank})")
    if response.answer:
        print(f"\nAnswer ({response.iterations} iteration(s)):\n{response.answer}")
    if response.trace:
        print(f"\nAgent trace: {' | '.join(response.trace)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
