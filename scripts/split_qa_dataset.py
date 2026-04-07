#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import typer

from _bootstrap import PROJECT_ROOT
from leaflet_pipeline.pipeline_building import assign_split
from leaflet_pipeline.pipeline_schema import QASample

app = typer.Typer(add_completion=False, help="Split QA dataset into train/test sets.")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset"


def iter_samples(input_path: Path) -> list[QASample]:
    samples: list[QASample] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(QASample.model_validate_json(line))
    return samples


@app.command()
def main(
    input_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Output directory for split JSONL files.",
    ),
    respect_split_hint: bool = typer.Option(
        True,
        help="Use the split_hint embedded in each QA sample when available.",
    ),
    train_ratio: float = typer.Option(0.9, min=0.0, max=1.0),
) -> None:
    if not 0.0 < train_ratio < 1.0:
        typer.echo("train_ratio must be between 0 and 1", err=True)
        raise typer.Exit(code=1)

    samples = iter_samples(input_path)
    if not samples:
        typer.echo("No QA samples found.", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    available_hints = {sample.split_hint for sample in samples if sample.split_hint}
    if respect_split_hint and available_hints:
        split_names = [name for name in ("train", "test", "val") if name in available_hints]
        if not split_names:
            split_names = ["train", "test"]
        split_paths = {name: output_dir / f"{name}.jsonl" for name in split_names}
        writers = {name: path.open("w", encoding="utf-8") for name, path in split_paths.items()}
        counts = {name: 0 for name in split_paths}
        try:
            for sample in samples:
                split_name = sample.split_hint if sample.split_hint in split_paths else "train"
                writers[split_name].write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
                counts[split_name] += 1
        finally:
            for handle in writers.values():
                handle.close()
        typer.echo(
            "Wrote split dataset from sample split_hint: "
            + ", ".join(f"{name}={counts[name]}" for name in split_paths)
            + f" under {output_dir}"
        )
        return

    split_paths = {
        "train": output_dir / "train.jsonl",
        "test": output_dir / "test.jsonl",
    }

    writers = {name: path.open("w", encoding="utf-8") for name, path in split_paths.items()}
    counts = {name: 0 for name in split_paths}
    try:
        for sample in samples:
            split_name = assign_split(sample.doc_id, train_ratio=train_ratio, val_ratio=0.0)
            writers[split_name].write(json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n")
            counts[split_name] += 1
    finally:
        for handle in writers.values():
            handle.close()

    typer.echo(
        "Wrote fallback split dataset by doc_id: "
        + ", ".join(f"{name}={counts[name]}" for name in ("train", "test"))
        + f" under {output_dir}"
    )
    empty_splits = [name for name, count in counts.items() if count == 0]
    if empty_splits:
        typer.echo(
            "Warning: empty split detected for "
            + ", ".join(empty_splits)
            + ". This can happen on very small samples."
        )


if __name__ == "__main__":
    app()
