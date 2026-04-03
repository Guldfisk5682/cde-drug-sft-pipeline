# CDE Drug SFT Pipeline Agent

## Project Goal

Build a small but complete data-to-SFT pipeline for Chinese drug leaflet QA construction based on publicly available CDE listed-drug records and leaflet PDFs.

The first milestone is to collect and process the first `500` orally administered drugs whose dosage forms are limited to:

- `片`
- `胶囊`
- `颗粒`

The downstream dataset will be a cleaned QA-style SFT corpus, with `用法用量` as the highest-priority field and LoRA or DoRA fine-tuning as the default training path.

## Why This Project

- Data source is public and authoritative.
- Leaflet PDFs are semi-structured, which makes them ideal for pipeline practice.
- The task is narrow enough for a student-scale project.
- The value can be shown through cleaning quality and field-grounded QA, not leaderboard chasing.

## Scope V0

### Sampling

- Source: CDE listed-drug catalog + public leaflet PDF attachments
- Max sample count: `500` drug leaflets
- Dosage forms: `片剂 / 胶囊 / 颗粒`
- Sampling order: use the first valid records that satisfy the filter and expose a usable leaflet PDF

### Target Fields

Required:

- `drug_name`
- `dosage_form`
- `spec`
- `otc_or_rx_flag`
- `indications`
- `dosage_and_administration`

Optional in V0:

- `adverse_reactions`

Candidate next fields:

- `contraindications`
- `precautions`
- `drug_interactions`

## Core Deliverables

1. A reproducible catalog-to-PDF ingestion pipeline
2. A normalized document schema for each leaflet
3. A cleaning pipeline for high-value fields, especially `用法用量`
4. A QA construction pipeline based on cleaned fields
5. A Dockerized runtime that can run the same way locally and on a remote GPU server
6. A baseline SFT stage built around `PyTorch + Transformers + Datasets`

## Non-Goals

- Building a medical diagnosis assistant
- Claiming state-of-the-art model performance
- Crawling the full CDE drug universe in V0
- Solving every noisy PDF edge case before the first end-to-end run

## Proposed Data Flow

1. Query or parse CDE catalog records
2. Filter records by dosage form and attachment availability
3. Save metadata into a raw catalog table
4. Download leaflet PDFs
5. Extract PDF text
6. Segment text into target sections
7. Normalize high-priority fields
8. Build field-grounded QA pairs
9. Export train/validation/test-ready JSONL
10. Launch downstream SFT experiments

## Data Schema

Each raw document should eventually map into a record like:

```json
{
  "doc_id": "cde-000001",
  "drug_name": "示例药品",
  "dosage_form": "片剂",
  "spec": "0.25g",
  "otc_or_rx_flag": "RX",
  "pdf_url": "https://...",
  "pdf_path": "data/raw/pdfs/cde-000001.pdf",
  "source_url": "https://...",
  "indications": "...",
  "dosage_and_administration": "...",
  "adverse_reactions": "...",
  "raw_text_path": "data/interim/text/cde-000001.txt"
}
```

## Cleaning Focus

`用法用量` is the hardest and most valuable field. The first cleaning rules should target:

- broken line merges
- duplicated section headers
- unit normalization such as `mg / g / ml`
- frequency phrases such as `一日X次`
- special population hints such as `儿童 / 老年人 / 孕妇`

## Evaluation Direction

The first benchmark should be document-grounded and narrow:

- dosage-and-administration QA
- indication QA
- prescription-vs-OTC recognition

Test splitting must happen at the document level, not the QA level.

## Training Direction

The training stage is intentionally simple compared with the data work:

- base models: `Qwen`-class `4B-9B`
- training method: `LoRA / DoRA`
- stack: `PyTorch`, `Transformers`, `Datasets`, `PEFT`
- runtime: Docker first, then the same image on a remote GPU server

## Repository Conventions

- `data/raw/`: untouched catalog files and original PDFs
- `data/interim/`: extracted text and intermediate parse artifacts
- `data/processed/`: normalized document JSONL and QA datasets
- `configs/`: sampling and runtime configs
- `scripts/`: runnable entrypoints
- `src/`: reusable Python package code
- `docs/`: notes, decisions, and benchmark drafts

## Immediate Next Steps

1. Implement CDE catalog ingestion for the target dosage forms
2. Confirm how leaflet attachment URLs are exposed in CDE pages
3. Build PDF downloader with retry and metadata logging
4. Prototype section extraction on `20-30` leaflets
5. Lock the first normalized schema before QA generation

## Working Principle

Prefer a narrow, reproducible, inspectable pipeline over a large but brittle crawl. Every stage should leave artifacts on disk so that failures can be debugged without rerunning the entire workflow.
