#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.benchmarking import build_benchmark_sample, iter_qa_samples, write_jsonl

app = typer.Typer(add_completion=False, help="Build a benchmark JSONL from held-out QA samples.")

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "dataset" / "test.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "benchmark" / "benchmark.jsonl"


@app.command()
def main(
    input_path: Path = typer.Option(DEFAULT_INPUT_PATH, exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_PATH),
    limit: int | None = typer.Option(None, min=1, help="Optional total sample limit."),
    per_task_limit: int | None = typer.Option(
        None, min=1, help="Optional per-task cap for a smaller first benchmark."
    ),
) -> None:
    qa_samples = iter_qa_samples(input_path)
    benchmark_samples = [build_benchmark_sample(sample) for sample in qa_samples]

    if per_task_limit is not None:
        grouped: dict[str, list] = defaultdict(list)
        for sample in benchmark_samples:
            if len(grouped[sample.task_type]) < per_task_limit:
                grouped[sample.task_type].append(sample)
        benchmark_samples = [sample for task in sorted(grouped) for sample in grouped[task]]

    if limit is not None:
        benchmark_samples = benchmark_samples[:limit]

    write_jsonl(output_path, benchmark_samples)
    typer.echo(f"Wrote {len(benchmark_samples)} benchmark samples to {output_path}")


if __name__ == "__main__":
    app()
