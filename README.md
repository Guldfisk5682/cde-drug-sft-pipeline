# CDE Drug SFT Pipeline

Student-scale end-to-end project for building a Chinese drug leaflet QA dataset from public CDE listed-drug records and leaflet PDFs, then running SFT on top of the cleaned corpus.

## Current Scope

- First `500` valid CDE drug records
- Dosage forms limited to `片剂 / 胶囊 / 颗粒`
- Priority fields:
  - `用法用量`
  - `适应症`
  - `处方/非处方标识`
  - `不良反应` as an optional field in V0
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

Install editable dependencies inside the container if needed:

```bash
pip install -e .
```

Check the PyTorch and CUDA runtime inside the container:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Expected Workflow

1. Use the browser-side userscript to capture CDE detail-page attachment manifests
2. Save exported session JSON files under `scripts/data/`
3. Run the shell downloader to fetch original leaflet PDFs into `data/pdf/`
4. Extract and clean text
5. Build normalized document records
6. Generate QA pairs
7. Export JSONL for SFT
8. Run LoRA or DoRA fine-tuning

## Repository Layout

- `data/pdf/`: raw leaflet PDF downloads
- `scripts/cde_leaflet_helper.user.js`: browser-side CDE helper script
- `scripts/download_leaflets.sh`: batch PDF downloader
- `scripts/data/`: exported browser session JSON manifests

## Browser-Assisted Collection

The current CDE frontend challenge blocks a simple direct crawl from reliably reaching leaflet attachments. V0 therefore uses a browser-assisted collection step:

- open CDE pages in Edge or Chrome
- use a Tampermonkey userscript to capture rendered list and detail pages
- export structured detail-page manifests into `scripts/data/`
- run the shell downloader to fetch original PDFs into `data/pdf/`
