#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.leaflet_cleaning import extract_leaflet_fields
from leaflet_pipeline.pipeline_building import assign_split, build_clean_record, build_qa_samples
from leaflet_pipeline.pipeline_schema import QASample

app = typer.Typer(add_completion=False, help="Run the full PDF -> cleaned record -> QA -> split pipeline.")

DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset"


def iter_pdf_paths(pdf_dir: Path, limit: int | None) -> list[Path]:
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if limit is not None:
        return pdf_paths[:limit]
    return pdf_paths


@app.command()
def main(
    pdf_dir: Path = typer.Option(..., exists=True, file_okay=False, readable=True),
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR),
    dataset_dir: Path = typer.Option(DEFAULT_DATASET_DIR),
    limit: int | None = typer.Option(None, min=1),
    valid_only: bool = typer.Option(True, help="Only keep cleaned records that pass required-field checks."),
    respect_split_hint: bool = typer.Option(True, help="Use QA sample split_hint when available."),
    train_ratio: float = typer.Option(0.9, min=0.0, max=1.0),
) -> None:
    pdf_paths = iter_pdf_paths(pdf_dir, limit)
    if not pdf_paths:
        typer.echo("No PDF files found.", err=True)
        raise typer.Exit(code=1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    field_path = processed_dir / "leaflet_fields.jsonl"
    clean_path = processed_dir / "leaflet_records.clean.jsonl"
    qa_path = dataset_dir / "qa_dataset.jsonl"
    train_path = dataset_dir / "train.jsonl"
    test_path = dataset_dir / "test.jsonl"
    val_path = dataset_dir / "val.jsonl"

    field_count = 0
    clean_count = 0
    qa_count = 0
    split_counts = {"train": 0, "test": 0, "val": 0}

    with (
        field_path.open("w", encoding="utf-8") as field_fh,
        clean_path.open("w", encoding="utf-8") as clean_fh,
        qa_path.open("w", encoding="utf-8") as qa_fh,
        train_path.open("w", encoding="utf-8") as train_fh,
        test_path.open("w", encoding="utf-8") as test_fh,
        val_path.open("w", encoding="utf-8") as val_fh,
    ):
        split_writers = {"train": train_fh, "test": test_fh, "val": val_fh}
        for pdf_path in pdf_paths:
            extraction = extract_leaflet_fields(pdf_path)
            field_fh.write(json.dumps(extraction.model_dump(mode="json"), ensure_ascii=False) + "\n")
            field_count += 1

            record = build_clean_record(extraction)
            if valid_only and not record.is_valid:
                continue
            clean_fh.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")
            clean_count += 1

            for sample in build_qa_samples(record):
                qa_fh.write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                qa_count += 1
                if respect_split_hint and sample.split_hint in split_writers:
                    split_name = sample.split_hint
                else:
                    split_name = assign_split(sample.doc_id, train_ratio=train_ratio, val_ratio=0.0)
                split_writers[split_name].write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                split_counts[split_name] += 1

    if split_counts["val"] == 0 and val_path.exists():
        val_path.unlink()

    typer.echo(
        "Pipeline finished: "
        f"fields={field_count}, cleaned={clean_count}, qa={qa_count}, "
        f"train={split_counts['train']}, test={split_counts['test']}, val={split_counts['val']}"
    )


if __name__ == "__main__":
    app()
