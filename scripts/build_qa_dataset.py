#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.pipeline_building import build_qa_samples
from leaflet_pipeline.pipeline_schema import CleanLeafletRecord

app = typer.Typer(add_completion=False, help="Build QA-style SFT samples from cleaned leaflet records.")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset"


def iter_clean_records(input_path: Path) -> list[CleanLeafletRecord]:
    records: list[CleanLeafletRecord] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(CleanLeafletRecord.model_validate_json(line))
    return records


@app.command()
def main(
    input_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "qa_dataset.jsonl",
        help="JSONL output path for QA-style SFT samples.",
    ),
) -> None:
    records = iter_clean_records(input_path)
    if not records:
        typer.echo("No cleaned records found.", err=True)
        raise typer.Exit(code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            for sample in build_qa_samples(record):
                fh.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                written += 1

    typer.echo(f"Wrote {written} QA samples to {output_path}")


if __name__ == "__main__":
    app()
