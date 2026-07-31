"""Run the end-to-end generation evaluation and emit a markdown report.

Runs the full pipeline (retrieve → build context → prompt → generate → map
citations) for every query in a dataset and writes a markdown diagnostics report
for manual inspection. Evaluation is qualitative — no automated answer scoring.

Run:
    python -m app.evaluation.run_generation [dataset.json] [--out report.md]

Set ``LLM_BACKEND=mock`` to run without a live LLM (useful for a smoke test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.dataset import EvaluationDataset
from app.evaluation.generation import evaluate_dataset
from app.evaluation.generation_report import format_report
from app.evaluation.paths import DEFAULT_DATASET_PATH
from app.services.answer_service import get_answer_service


def main(dataset_path: Path, out_path: Path | None = None) -> None:
    dataset = EvaluationDataset.from_file(dataset_path)
    service = get_answer_service()

    evaluations = evaluate_dataset(service, dataset.cases)
    report = format_report(evaluations)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote generation report for {len(evaluations)} queries to {out_path}")
    else:
        print(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the evaluation dataset JSON (defaults to the golden dataset).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the markdown report to this path instead of stdout.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        Path(args.dataset),
        Path(args.out) if args.out else None,
    )
