# CDE Drug SFT Pipeline

Student-scale end-to-end project for building a Chinese drug leaflet QA dataset from public CDE listed-drug records and leaflet PDFs, then running SFT on top of the cleaned corpus.

## Current Scope

- First `500` valid CDE drug records
- Dosage forms limited to `片剂 / 胶囊 / 颗粒`
- Priority fields:
  - `用法用量`
  - `适应症`
  - `处方/非处方标识`
  - `禁忌/警告`
- Training stack:
  - `PyTorch`
  - `Transformers`
  - `Datasets`
  - `PEFT`

## Quick Start

Build the container:

```bash
docker compose build
```

Open the project shell inside the container:

```bash
docker compose run --rm app bash
```

Check the PyTorch and CUDA runtime inside the container:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Expected Workflow

1. Use the browser-side userscript to capture CDE detail-page attachment manifests
2. Save exported session JSON files under `scripts/data/`
3. Run the shell downloader to fetch original leaflet PDFs into `data/pdf/`
4. Extract first-pass fields from PDFs
5. Clean extracted fields into normalized document records
6. Generate QA pairs from cleaned records
7. Split JSONL into train/test sets
8. Run LoRA or DoRA fine-tuning

## Repository Layout

- `data/pdf/`: raw leaflet PDF downloads
- `data/interim/`: optional intermediate parsing artifacts
- `data/processed/`: cleaned document-level records
- `data/dataset/`: QA datasets and split outputs
- `scripts/cde_leaflet_helper.user.js`: browser-side CDE helper script
- `scripts/download_leaflets.sh`: batch PDF downloader
- `scripts/data/`: exported browser session JSON manifests

## Staged Pipeline

The current local-first pipeline is intentionally explicit:

1. `extract_fields`
   - input: raw PDFs under `data/pdf/<batch>/`
   - output: `data/processed/leaflet_fields.jsonl`
2. `clean_records`
   - input: extracted field JSONL
   - output: `data/processed/leaflet_records.clean.jsonl`
3. `build_qa`
   - input: cleaned document records
   - output: `data/dataset/qa_dataset.jsonl`
4. `split_dataset`
   - input: QA dataset
   - output: `data/dataset/{train,test}.jsonl`

Example commands:

```bash
python scripts/extract_leaflet_fields.py \
  --pdf-dir data/pdf/20260404T151727Z \
  --output-path data/processed/leaflet_fields.sample.jsonl \
  --limit 20

python scripts/clean_leaflet_records.py \
  --input-path data/processed/leaflet_fields.sample.jsonl \
  --output-path data/processed/leaflet_records.clean.sample.jsonl

python scripts/build_qa_dataset.py \
  --input-path data/processed/leaflet_records.clean.sample.jsonl \
  --output-path data/dataset/qa_dataset.sample.jsonl

python scripts/split_qa_dataset.py \
  --input-path data/dataset/qa_dataset.sample.jsonl \
  --output-dir data/dataset/sample_split
```

Training uses `configs/train.yaml` by default. The current config keeps `bf16`
enabled and leaves `test.jsonl` out of online evaluation so the held-out test
split is not consumed during training.

## Browser-Assisted Collection

The current CDE frontend challenge blocks a simple direct crawl from reliably reaching leaflet attachments. V0 therefore uses a browser-assisted collection step:

- open CDE pages in Edge or Chrome
- use a Tampermonkey userscript to capture rendered list and detail pages
- export structured detail-page manifests into `scripts/data/`
- run the shell downloader to fetch original PDFs into `data/pdf/`
