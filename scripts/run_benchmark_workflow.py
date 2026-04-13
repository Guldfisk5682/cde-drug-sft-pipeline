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
    help="Build benchmark data, run base/SFT generation, evaluate both, and compare summaries.",
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "dataset" / "test.jsonl"
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"


def run_step(title: str, command: list[str]) -> None:
    print(f"[workflow] {title}", flush=True)
    print("[workflow] command: " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


@app.command()
def main(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, exists=True, dir_okay=False, readable=True
    ),
    test_dataset_path: Path = typer.Option(
        DEFAULT_TEST_PATH, exists=True, dir_okay=False, readable=True
    ),
    benchmark_dir: Path = typer.Option(DEFAULT_BENCHMARK_DIR),
    base_model_name_or_path: str = typer.Option(..., help="Base model hub id or local path."),
    sft_adapter_path: Path = typer.Option(..., exists=True, dir_okay=True, readable=True),
    benchmark_limit: int | None = typer.Option(None, min=1),
    per_task_limit: int | None = typer.Option(None, min=1),
    generation_limit: int | None = typer.Option(None, min=1),
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = benchmark_dir / "benchmark.jsonl"
    base_predictions_path = benchmark_dir / "base_predictions.jsonl"
    sft_predictions_path = benchmark_dir / "sft_predictions.jsonl"
    base_results_path = benchmark_dir / "base_eval.results.jsonl"
    sft_results_path = benchmark_dir / "sft_eval.results.jsonl"
    base_summary_path = benchmark_dir / "base_eval.summary.json"
    sft_summary_path = benchmark_dir / "sft_eval.summary.json"
    comparison_json_path = benchmark_dir / "comparison.summary.json"
    comparison_txt_path = benchmark_dir / "comparison.summary.txt"

    build_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_benchmark_dataset.py"),
        "--input-path",
        str(test_dataset_path),
        "--output-path",
        str(benchmark_path),
    ]
    if benchmark_limit is not None:
        build_command.extend(["--limit", str(benchmark_limit)])
    if per_task_limit is not None:
        build_command.extend(["--per-task-limit", str(per_task_limit)])
    run_step("Building benchmark dataset", build_command)

    base_generation_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_benchmark_generation.py"),
        "--config-path",
        str(config_path),
        "--benchmark-path",
        str(benchmark_path),
        "--output-path",
        str(base_predictions_path),
        "--model-name-or-path",
        base_model_name_or_path,
    ]
    if generation_limit is not None:
        base_generation_command.extend(["--limit", str(generation_limit)])
    run_step("Running base-model benchmark generation", base_generation_command)

    sft_generation_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_benchmark_generation.py"),
        "--config-path",
        str(config_path),
        "--benchmark-path",
        str(benchmark_path),
        "--output-path",
        str(sft_predictions_path),
        "--model-name-or-path",
        base_model_name_or_path,
        "--adapter-path",
        str(sft_adapter_path),
    ]
    if generation_limit is not None:
        sft_generation_command.extend(["--limit", str(generation_limit)])
    run_step("Running SFT benchmark generation", sft_generation_command)

    run_step(
        "Evaluating base-model predictions",
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_benchmark.py"),
            "--benchmark-path",
            str(benchmark_path),
            "--predictions-path",
            str(base_predictions_path),
            "--results-path",
            str(base_results_path),
            "--summary-path",
            str(base_summary_path),
        ],
    )
    run_step(
        "Evaluating SFT predictions",
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_benchmark.py"),
            "--benchmark-path",
            str(benchmark_path),
            "--predictions-path",
            str(sft_predictions_path),
            "--results-path",
            str(sft_results_path),
            "--summary-path",
            str(sft_summary_path),
        ],
    )
    run_step(
        "Comparing benchmark summaries",
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
                "benchmark_path": str(benchmark_path),
                "base_predictions_path": str(base_predictions_path),
                "sft_predictions_path": str(sft_predictions_path),
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
