#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Compare two benchmark evaluation summary JSON files.")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_or_none(bucket: dict, metric: str) -> float | None:
    value = bucket.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return None


@app.command()
def main(
    base_summary_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    sft_summary_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
) -> None:
    base = load_json(base_summary_path)
    sft = load_json(sft_summary_path)
    metrics = (
        "classification_accuracy",
        "avg_recall",
        "avg_precision",
        "avg_hallucination_rate",
        "exact_match_rate",
    )

    typer.echo("Overall:")
    for metric in metrics:
        base_value = metric_or_none(base.get("overall", {}), metric)
        sft_value = metric_or_none(sft.get("overall", {}), metric)
        if base_value is None and sft_value is None:
            continue
        delta = None if base_value is None or sft_value is None else sft_value - base_value
        typer.echo(
            f"  {metric}: base={base_value} sft={sft_value} delta={delta}"
        )

    all_tasks = sorted(
        set(base.get("by_task_type", {}).keys()) | set(sft.get("by_task_type", {}).keys())
    )
    typer.echo("\nBy task type:")
    for task in all_tasks:
        typer.echo(f"  {task}:")
        base_bucket = base.get("by_task_type", {}).get(task, {})
        sft_bucket = sft.get("by_task_type", {}).get(task, {})
        for metric in metrics:
            base_value = metric_or_none(base_bucket, metric)
            sft_value = metric_or_none(sft_bucket, metric)
            if base_value is None and sft_value is None:
                continue
            delta = None if base_value is None or sft_value is None else sft_value - base_value
            typer.echo(f"    {metric}: base={base_value} sft={sft_value} delta={delta}")


if __name__ == "__main__":
    app()
