# Scripts

Planned entrypoints:

- `fetch_cde_catalog.py`: collect target catalog records from CDE
- `download_leaflets.py`: download leaflet PDFs and write metadata logs
- `extract_sections.py`: parse PDF text and split target sections
- `build_qa.py`: convert cleaned fields into QA JSONL
- `train_sft.py`: run LoRA or DoRA fine-tuning on exported QA data
- `eval_sft.py`: run task-specific evaluation on held-out documents
