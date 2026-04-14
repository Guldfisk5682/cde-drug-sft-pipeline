#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.benchmark_schema import BenchmarkSampleResult
from leaflet_pipeline.benchmarking import (
    evaluate_prediction,
    iter_benchmark_predictions,
    iter_benchmark_samples,
    summarize_results,
    write_jsonl,
)
from llm_judge.deepseek_judge import load_dotenv, request_judgment

app = typer.Typer(add_completion=False, help="Run DeepSeek-based LLM judge evaluation on benchmark outputs.")

DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "benchmark" / "predictions.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "benchmark" / "llm_judge.results.jsonl"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "benchmark" / "llm_judge.summary.json"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


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
    env_path: Path = typer.Option(DEFAULT_ENV_PATH, dir_okay=False),
    judge_model: str = typer.Option("deepseek-chat"),
    base_url: str | None = typer.Option(None),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    load_dotenv(env_path)
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        typer.echo("Missing DEEPSEEK_API_KEY in environment or .env file.", err=True)
        raise typer.Exit(code=1)
    if base_url is None:
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    benchmark_samples = iter_benchmark_samples(benchmark_path)
    predictions = iter_benchmark_predictions(predictions_path)
    sample_by_id = {sample.benchmark_id: sample for sample in benchmark_samples}
    prediction_by_id = {prediction.benchmark_id: prediction for prediction in predictions}

    benchmark_ids = [sample.benchmark_id for sample in benchmark_samples if sample.benchmark_id in prediction_by_id]
    if limit is not None:
        benchmark_ids = benchmark_ids[:limit]

    results: list[BenchmarkSampleResult] = []
    for index, benchmark_id in enumerate(benchmark_ids, start=1):
        sample = sample_by_id[benchmark_id]
        prediction = prediction_by_id[benchmark_id]
        if sample.answer_type == "classification":
            result = evaluate_prediction(sample, prediction.prediction)
            result.predicted_slots = {"judge_mode": ["rule"]}
            results.append(result)
            continue

        print(
            f"[llm-judge] sample {index}/{len(benchmark_ids)} "
            f"task={sample.task_type} benchmark_id={benchmark_id}",
            flush=True,
        )
        judgment = request_judgment(
            question=sample.question,
            gold_answer=sample.reference_answer,
            prediction=prediction.prediction,
            model=judge_model,
            api_key=api_key,
            base_url=base_url,
        )
        results.append(
            BenchmarkSampleResult(
                "benchmark_id": sample.benchmark_id,
                task_type=sample.task_type,
                answer_type=sample.answer_type,
                question=sample.question,
                reference_answer=sample.reference_answer,
                prediction=prediction.prediction,
                gold_label=sample.gold_label,
                predicted_label=None,
                label_correct=None,
                gold_slots=sample.gold_slots,
                predicted_slots={"judge_mode": ["llm"], "judge_reason": [judgment.get("reason", "")]},
                matched_values=judgment.get("matched_points", []),
                missing_values=judgment.get("missing_points", []),
                hallucinated_values=judgment.get("hallucinated_points", []),
                recall=float(judgment.get("recall_score", 0.0)),
                precision=float(judgment.get("precision_score", 0.0)),
                hallucination_rate=float(judgment.get("hallucination_score", 0.0)),
                exact_match=False,
            )
        )

    summary = summarize_results(results)
    summary["benchmark_path"] = str(benchmark_path)
    summary["predictions_path"] = str(predictions_path)
    summary["sample_count"] = len(results)
    summary["judge_model"] = judge_model
    summary["judge_mode"] = "hybrid_rule_plus_llm"

    write_jsonl(results_path, results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"Wrote {len(results)} LLM-judge evaluation rows to {results_path}")
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
