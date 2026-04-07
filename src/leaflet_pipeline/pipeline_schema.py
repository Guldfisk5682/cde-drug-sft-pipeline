from __future__ import annotations

from pydantic import BaseModel, Field


class CleanLeafletRecord(BaseModel):
    doc_id: str
    source_pdf_path: str
    drug_name: str | None = None
    otc_or_rx_flag: str = "UNKNOWN"
    indications: str | None = None
    dosage_and_administration: str | None = None
    contraindications: str | None = None
    warnings: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    is_valid: bool = False
    missing_fields: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class QASample(BaseModel):
    sample_id: str
    doc_id: str
    drug_name: str
    task_type: str
    answer_field: str
    template_id: str
    split_hint: str = "train"
    chunk_index: int | None = None
    messages: list[ChatMessage]
