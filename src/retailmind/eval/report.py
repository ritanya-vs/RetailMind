"""Save and print the outcome of one evaluation run."""

from __future__ import annotations

import json
from pathlib import Path

from retailmind.eval.metrics import Results


def save_run(out_dir: Path, report: dict, by_query: dict, results: Results) -> None:
    """Write metrics.json (summary), per_query.json (for significance tests) and run.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "per_query.json").write_text(json.dumps(by_query), encoding="utf-8")
    run = {query_id: dict(hits) for query_id, hits in results.items()}
    (out_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")


def print_metrics(title: str, means: dict[str, float]) -> None:
    print(title)
    for name, value in means.items():
        print(f"  {name:15s} {value:.4f}")
