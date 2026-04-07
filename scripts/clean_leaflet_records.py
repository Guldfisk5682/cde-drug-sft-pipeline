#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.leaflet_schema import LeafletExtraction
from leaflet_pipeline.pipeline_building import build_clean_record

app = typer.Typer(add_completion=False, help="Normalize extracted leaflet fields into cleaned document records.")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def iter_extractions(input_path: Path) -> list[LeafletExtraction]:
    records: list[LeafletExtraction] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(LeafletExtraction.model_validate_json(line))
    return records


@app.command()
def main(
    input_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "leaflet_records.clean.jsonl",
        help="JSONL output path for cleaned document records.",
    ),
    valid_only: bool = typer.Option(False, help="Only keep records that pass the required-field checks."),
) -> None:
    extractions = iter_extractions(input_path)
    if not extractions:
        typer.echo("No extracted records found.", err=True)
        raise typer.Exit(code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for extraction in extractions:
            record = build_clean_record(extraction)
            if valid_only and not record.is_valid:
                continue
            fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
            written += 1

    typer.echo(f"Wrote {written} cleaned records to {output_path}")


if __name__ == "__main__":
    app()
