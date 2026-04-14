#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT

app = typer.Typer(
    add_completion=False,
    help="Run DeepSeek LLM judge on both base and SFT predictions, then compare summaries.",
)

DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def run_step(title: str, command: list[str]) -> None:
    print(f"[llm-judge-workflow] {title}", flush=True)
    print("[llm-judge-workflow] command: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


@app.command()
def main(
    benchmark_dir: Path = typer.Option(DEFAULT_BENCHMARK_DIR, exists=True, file_okay=False, readable=True),
    env_path: Path = typer.Option(DEFAULT_ENV_PATH, dir_okay=False),
    judge_model: str = typer.Option("deepseek-chat"),
    limit: int | None = typer.Option(None, min=1),
) -> None:
    benchmark_path = benchmark_dir / "benchmark.jsonl"
    base_predictions_path = benchmark_dir / "base_predictions.jsonl"
    sft_predictions_path = benchmark_dir / "sft_predictions.jsonl"
    base_results_path = benchmark_dir / "base_llm_judge.results.jsonl"
    sft_results_path = benchmark_dir / "sft_llm_judge.results.jsonl"
    base_summary_path = benchmark_dir / "base_llm_judge.summary.json"
    sft_summary_path = benchmark_dir / "sft_llm_judge.summary.json"
    comparison_json_path = benchmark_dir / "llm_judge_comparison.summary.json"
    comparison_txt_path = benchmark_dir / "llm_judge_comparison.summary.txt"

    base_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_llm_judge_evaluation.py"),
        "--benchmark-path",
        str(benchmark_path),
        "--predictions-path",
        str(base_predictions_path),
        "--results-path",
        str(base_results_path),
        "--summary-path",
        str(base_summary_path),
        "--env-path",
        str(env_path),
        "--judge-model",
        judge_model,
    ]
    if limit is not None:
        base_command.extend(["--limit", str(limit)])
    run_step("Evaluating base-model predictions with LLM judge", base_command)

    sft_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_llm_judge_evaluation.py"),
        "--benchmark-path",
        str(benchmark_path),
        "--predictions-path",
        str(sft_predictions_path),
        "--results-path",
        str(sft_results_path),
        "--summary-path",
        str(sft_summary_path),
        "--env-path",
        str(env_path),
        "--judge-model",
        judge_model,
    ]
    if limit is not None:
        sft_command.extend(["--limit", str(limit)])
    run_step("Evaluating SFT predictions with LLM judge", sft_command)

    run_step(
        "Comparing LLM-judge summaries",
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compare_benchmark_runs.py"),
            "--base-summary-path",
            str(base_summary_path),
            "--sft-summary-path",
            str(sft_summary_path),
            "--output-json-path",
            str(comparison_json_path),
            "--output-txt-path",
            str(comparison_txt_path),
        ],
    )

    print(
        json.dumps(
            {
                "base_summary_path": str(base_summary_path),
                "sft_summary_path": str(sft_summary_path),
                "comparison_json_path": str(comparison_json_path),
                "comparison_txt_path": str(comparison_txt_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    app()
