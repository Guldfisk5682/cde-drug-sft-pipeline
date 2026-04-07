#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

import typer

app = typer.Typer(add_completion=False, help="Evaluate extracted leaflet field coverage from a JSONL file.")

CORE_FIELDS = (
    "otc_or_rx_flag",
    "indications",
    "dosage_and_administration",
    "contraindications",
)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def summarize_lengths(records: list[dict], field_name: str) -> dict[str, float] | None:
    lengths = [len(record[field_name].strip()) for record in records if has_text(record.get(field_name))]
    if not lengths:
        return None
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": round(mean(lengths), 1),
    }


@app.command()
def main(
    input_path: Path = typer.Option(..., exists=True, dir_okay=False, readable=True, help="JSONL file produced by extract_leaflet_fields.py"),
    show_missing: int = typer.Option(10, min=0, help="How many missing examples to print for each field."),
) -> None:
    records = load_jsonl(input_path)
    if not records:
        typer.echo("No records found.", err=True)
        raise typer.Exit(code=1)

    total = len(records)
    warning_counter = Counter()
    for record in records:
        warning_counter.update(record.get("warnings", []))

    typer.echo(f"Input: {input_path}")
    typer.echo(f"Total records: {total}")

    otc_hits = sum(record.get("otc_or_rx_flag") not in (None, "", "UNKNOWN") for record in records)
    typer.echo(f"otc_or_rx_flag: {otc_hits}/{total}")

    for field_name in CORE_FIELDS[1:]:
        hits = sum(has_text(record.get(field_name)) for record in records)
        typer.echo(f"{field_name}: {hits}/{total}")

    typer.echo("")
    typer.echo("Length summary:")
    for field_name in CORE_FIELDS[1:]:
        summary = summarize_lengths(records, field_name)
        if summary is None:
            typer.echo(f"- {field_name}: no non-empty values")
            continue
        typer.echo(
            f"- {field_name}: min={summary['min']}, avg={summary['avg']}, max={summary['max']}"
        )

    if warning_counter:
        typer.echo("")
        typer.echo("Warnings:")
        for warning, count in warning_counter.most_common():
            typer.echo(f"- {warning}: {count}")

    if show_missing > 0:
        typer.echo("")
        typer.echo("Missing examples:")
        for field_name in CORE_FIELDS:
            missing = []
            for record in records:
                value = record.get(field_name)
                is_missing = value in (None, "", "UNKNOWN") if field_name == "otc_or_rx_flag" else not has_text(value)
                if is_missing:
                    missing.append(
                        {
                            "doc_id": record.get("doc_id"),
                            "drug_name_guess": record.get("drug_name_guess"),
                            "pdf_path": record.get("pdf_path"),
                            "warnings": record.get("warnings", []),
                        }
                    )
            typer.echo(f"- {field_name}: {len(missing)} missing")
            for item in missing[:show_missing]:
                typer.echo(
                    f"  doc_id={item['doc_id']} drug={item['drug_name_guess']} pdf={item['pdf_path']} warnings={item['warnings']}"
                )


if __name__ == "__main__":
    app()
