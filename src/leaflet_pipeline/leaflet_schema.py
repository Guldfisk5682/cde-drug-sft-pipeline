from __future__ import annotations

from pydantic import BaseModel, Field


class LeafletSection(BaseModel):
    heading: str
    text: str
    start_line: int | None = None
    end_line: int | None = None


class LeafletExtraction(BaseModel):
    doc_id: str
    pdf_path: str
    drug_name_guess: str | None = None
    otc_or_rx_flag: str = "UNKNOWN"
    indications: str | None = None
    dosage_and_administration: str | None = None
    contraindications: str | None = None
    sections: dict[str, LeafletSection] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
