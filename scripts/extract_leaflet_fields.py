#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.leaflet_cleaning import extract_leaflet_fields

app = typer.Typer(add_completion=False, help="Extract first-pass fields from leaflet PDFs.")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def iter_pdf_paths(pdf_dir: Path, limit: int | None) -> list[Path]:
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if limit is not None:
        return pdf_paths[:limit]
    return pdf_paths


@app.command()
def main(
    pdf_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "leaflet_fields.jsonl",
        help="JSONL output path for extracted and cleaned first-pass fields.",
    ),
    limit: int | None = typer.Option(
        None,
        min=1,
        help="Optional limit for dry runs on a small subset of PDFs.",
    ),
) -> None:
    pdf_paths = iter_pdf_paths(pdf_dir, limit)
    if not pdf_paths:
        typer.echo("No PDF files found.", err=True)
        raise typer.Exit(code=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for pdf_path in pdf_paths:
            extraction = extract_leaflet_fields(pdf_path)
            fh.write(json.dumps(extraction.model_dump(mode="json"), ensure_ascii=False) + "\n")
            count += 1

    typer.echo(f"Wrote {count} extracted records to {output_path}")


if __name__ == "__main__":
    app()
