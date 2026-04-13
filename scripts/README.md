# Scripts

Current active entrypoints:

- `cde_leaflet_helper.user.js`: browser-side helper for semi-automatic list/detail capture inside a real CDE session
- `download_leaflets.sh`: curl-based batch downloader that reads manifest JSON files from `scripts/data/`
- `extract_leaflet_fields.py`: first-pass PDF text extraction, section detection, and field cleaning
- `evaluate_leaflet_fields.py`: summarize extraction hit rate, missing fields, and warning counts from a JSONL output
- `clean_leaflet_records.py`: normalize extracted fields into cleaned document-level records
- `build_qa_dataset.py`: build QA-style SFT samples from cleaned records
- `split_qa_dataset.py`: split QA samples into train/test JSONL, preferring each sample's `split_hint`
- `run_pipeline.py`: one-shot PDF -> fields -> cleaned records -> QA dataset -> split dataset
- `train_qwen_dora.py`: Qwen3.5-4B DoRA training entrypoint for the finalized train/test JSONL
- `build_benchmark_dataset.py`: convert held-out `test.jsonl` into a benchmark JSONL with gold labels / slots
- `run_benchmark_generation.py`: run base-model or SFT-adapter generation on the benchmark
- `evaluate_benchmark.py`: score predictions with classification accuracy plus slot-based recall / precision / hallucination
- `compare_benchmark_runs.py`: compare two evaluation summaries, e.g. base model vs SFT model
- `run_benchmark_workflow.py`: one-shot benchmark builder + base/SFT generation + evaluation + comparison

Support directory:

- `data/`: exported browser session JSON manifests

Exported manifest shape:

- `detail_pages[].url`: detail page URL used as download referer
- `detail_pages[].attachment_links[].href`: direct attachment URL
- `detail_pages[].attachment_links[].file_name`: suggested PDF file name

First-pass extraction example:

```bash
python scripts/extract_leaflet_fields.py \
  --pdf-dir data/pdf/20260404T151727Z \
  --output-path data/processed/leaflet_fields.sample.jsonl \
  --limit 20
```

Evaluation example:

```bash
python scripts/evaluate_leaflet_fields.py \
  --input-path data/processed/leaflet_fields.sample.jsonl \
  --show-missing 5
```

Current priority fields:

- `otc_or_rx_flag`
- `indications`
- `dosage_and_administration`
- `contraindications` extracted from either `【禁忌】` or `【警告】`

Pipeline examples:

```bash
python scripts/run_pipeline.py \
  --pdf-dir data/pdf/20260404T151727Z

python scripts/clean_leaflet_records.py \
  --input-path data/processed/leaflet_fields.jsonl \
  --output-path data/processed/leaflet_records.clean.jsonl

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

Training example:

```bash
python scripts/train_qwen_dora.py \
  --config-path configs/train.yaml
```

The default training config uses `bf16` and keeps `eval_dataset_path: null`,
so `data/dataset/test.jsonl` remains a held-out test split rather than an
online evaluation set during training.

Current finalized dataset files:

- `data/processed/leaflet_fields.jsonl`
- `data/processed/leaflet_records.clean.jsonl`
- `data/dataset/qa_dataset.jsonl`
- `data/dataset/train.jsonl`
- `data/dataset/test.jsonl`

Benchmark workflow example:

```bash
python scripts/build_benchmark_dataset.py \
  --input-path data/dataset/test.jsonl \
  --output-path data/benchmark/benchmark.sample.jsonl \
  --per-task-limit 20

python scripts/run_benchmark_generation.py \
  --config-path configs/train.yaml \
  --benchmark-path data/benchmark/benchmark.sample.jsonl \
  --output-path data/benchmark/base_predictions.jsonl \
  --model-name-or-path Qwen/Qwen3.5-4B \
  --device-map cuda:0

python scripts/run_benchmark_generation.py \
  --config-path configs/train.yaml \
  --benchmark-path data/benchmark/benchmark.sample.jsonl \
  --output-path data/benchmark/sft_predictions.jsonl \
  --model-name-or-path Qwen/Qwen3.5-4B \
  --adapter-path outputs/qwen3.5-4b-dora/final \
  --device-map cuda:0

python scripts/evaluate_benchmark.py \
  --benchmark-path data/benchmark/benchmark.sample.jsonl \
  --predictions-path data/benchmark/base_predictions.jsonl \
  --results-path data/benchmark/base_eval.results.jsonl \
  --summary-path data/benchmark/base_eval.summary.json

python scripts/evaluate_benchmark.py \
  --benchmark-path data/benchmark/benchmark.sample.jsonl \
  --predictions-path data/benchmark/sft_predictions.jsonl \
  --results-path data/benchmark/sft_eval.results.jsonl \
  --summary-path data/benchmark/sft_eval.summary.json

python scripts/compare_benchmark_runs.py \
  --base-summary-path data/benchmark/base_eval.summary.json \
  --sft-summary-path data/benchmark/sft_eval.summary.json
```

The benchmark runner now supports batched generation. The initial default is
`benchmark.batch_size: 4` in `configs/train.yaml`, chosen conservatively for a
`Qwen3.5-4B` 8-bit model after observing roughly `5-6 GB` VRAM usage in single-sample inference.

One-shot workflow example:

```bash
python scripts/run_benchmark_workflow.py \
  --config-path configs/train.yaml \
  --test-dataset-path data/dataset/test.jsonl \
  --benchmark-dir data/benchmark/run01 \
  --base-model-name-or-path Qwen/Qwen3.5-4B \
  --sft-adapter-path outputs/qwen3.5-4b-dora/final \
  --per-task-limit 20
```
