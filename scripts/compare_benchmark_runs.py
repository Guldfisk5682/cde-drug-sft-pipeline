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


def build_comparison(base: dict, sft: dict) -> dict:
    metrics = (
        "classification_accuracy",
        "avg_recall",
        "avg_precision",
        "avg_hallucination_rate",
        "exact_match_rate",
    )
    comparison: dict[str, object] = {
        "base_summary_path": base.get("predictions_path"),
        "sft_summary_path": sft.get("predictions_path"),
        "overall": {},
        "by_task_type": {},
    }

    for metric in metrics:
        base_value = metric_or_none(base.get("overall", {}), metric)
        sft_value = metric_or_none(sft.get("overall", {}), metric)
        comparison["overall"][metric] = {
            "base": base_value,
            "sft": sft_value,
            "delta": None if base_value is None or sft_value is None else sft_value - base_value,
        }

    all_tasks = sorted(
        set(base.get("by_task_type", {}).keys()) | set(sft.get("by_task_type", {}).keys())
    )
    for task in all_tasks:
        base_bucket = base.get("by_task_type", {}).get(task, {})
        sft_bucket = sft.get("by_task_type", {}).get(task, {})
        task_metrics: dict[str, dict[str, float | None]] = {}
        for metric in metrics:
            base_value = metric_or_none(base_bucket, metric)
            sft_value = metric_or_none(sft_bucket, metric)
            task_metrics[metric] = {
                "base": base_value,
                "sft": sft_value,
                "delta": None if base_value is None or sft_value is None else sft_value - base_value,
            }
        comparison["by_task_type"][task] = task_metrics
    return comparison


def render_comparison_text(comparison: dict) -> str:
    lines: list[str] = ["Overall:"]
    for metric, payload in comparison.get("overall", {}).items():
        lines.append(
            f"  {metric}: base={payload['base']} sft={payload['sft']} delta={payload['delta']}"
        )

    lines.append("")
    lines.append("By task type:")
    for task, metrics in comparison.get("by_task_type", {}).items():
        lines.append(f"  {task}:")
        for metric, payload in metrics.items():
            lines.append(
                f"    {metric}: base={payload['base']} sft={payload['sft']} delta={payload['delta']}"
            )
    return "\n".join(lines)


@app.command()
def main(
    base_summary_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    sft_summary_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_json_path: Path | None = typer.Option(None),
    output_txt_path: Path | None = typer.Option(None),
) -> None:
    base = load_json(base_summary_path)
    sft = load_json(sft_summary_path)
    comparison = build_comparison(base, sft)
    rendered = render_comparison_text(comparison)

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_txt_path is not None:
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        output_txt_path.write_text(rendered + "\n", encoding="utf-8")

    typer.echo(rendered)


if __name__ == "__main__":
    app()
