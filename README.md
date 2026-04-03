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

1. Fetch catalog metadata
2. Filter records
3. Download leaflet PDFs
4. Extract and clean text
5. Build normalized document records
6. Generate QA pairs
7. Export JSONL for SFT
8. Run LoRA or DoRA fine-tuning
