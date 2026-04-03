# Decisions

## 2026-04-03

- Use CDE public data as the primary source
- Limit V0 to the first 500 valid records
- Restrict dosage forms to tablets, capsules, and granules
- Prioritize `用法用量`, `适应症`, and prescription/OTC detection
- Use Docker as the default runtime to avoid host Python drift
- Use the official PyTorch CUDA 12.1 image as the default container base
- Treat the repository as an end-to-end data-to-SFT project rather than a data-only pipeline
- Treat CDE collection as browser-assisted in V0 because direct HTTP crawling does not currently reach rendered catalog rows or leaflet attachment links
- Keep the repository structure minimal during collection: exported manifests under `scripts/data/` and original PDFs under `data/pdf/`
