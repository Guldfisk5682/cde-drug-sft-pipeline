#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.benchmarking import (
    evaluate_prediction,
    iter_benchmark_predictions,
    iter_benchmark_samples,
    summarize_results,
    write_jsonl,
)

app = typer.Typer(add_completion=False, help="Evaluate benchmark predictions against the gold benchmark.")

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "benchmark" / "predictions.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "benchmark" / "evaluation.results.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "benchmark" / "evaluation.summary.json"


@app.command()
def main(
    benchmark_path: Path = typer.Option(
        DEFAULT_BENCHMARK_PATH, exists=True, dir_okay=False, readable=True
    ),
    predictions_path: Path = typer.Option(
        DEFAULT_PREDICTIONS_PATH, exists=True, dir_okay=False, readable=True
    ),
    results_path: Path = typer.Option(DEFAULT_RESULTS_PATH),
    summary_path: Path = typer.Option(DEFAULT_SUMMARY_PATH),
) -> None:
    benchmark_samples = iter_benchmark_samples(benchmark_path)
    predictions = iter_benchmark_predictions(predictions_path)

    sample_by_id = {sample.benchmark_id: sample for sample in benchmark_samples}
    prediction_by_id = {prediction.benchmark_id: prediction for prediction in predictions}

    missing_predictions = [sample_id for sample_id in sample_by_id if sample_id not in prediction_by_id]
    if missing_predictions:
        typer.echo(
            f"Missing predictions for {len(missing_predictions)} benchmark samples; first few: "
            + ", ".join(missing_predictions[:5]),
            err=True,
        )
        raise typer.Exit(code=1)

    results = [
        evaluate_prediction(sample_by_id[benchmark_id], prediction_by_id[benchmark_id].prediction)
        for benchmark_id in sample_by_id
    ]
    summary = summarize_results(results)
    summary["benchmark_path"] = str(benchmark_path)
    summary["predictions_path"] = str(predictions_path)
    summary["sample_count"] = len(results)

    write_jsonl(results_path, results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"Wrote {len(results)} evaluation rows to {results_path}")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
