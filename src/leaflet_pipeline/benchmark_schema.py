from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkSample(BaseModel):
    benchmark_id: str
    source_sample_id: str
    doc_id: str
    drug_name: str
    task_type: str
    question: str
    reference_answer: str
    answer_type: str
    gold_label: str | None = None
    gold_slots: dict[str, list[str]] = Field(default_factory=dict)
    template_id: str | None = None
    chunk_index: int | None = None


class BenchmarkPrediction(BaseModel):
    benchmark_id: str
    model_name: str
    question: str
    prediction: str
    reference_answer: str
    task_type: str


class BenchmarkSampleResult(BaseModel):
    benchmark_id: str
    task_type: str
    answer_type: str
    question: str
    reference_answer: str
    prediction: str
    gold_label: str | None = None
    predicted_label: str | None = None
    label_correct: bool | None = None
    gold_slots: dict[str, list[str]] = Field(default_factory=dict)
    predicted_slots: dict[str, list[str]] = Field(default_factory=dict)
    matched_values: list[str] = Field(default_factory=list)
    missing_values: list[str] = Field(default_factory=list)
    hallucinated_values: list[str] = Field(default_factory=list)
    recall: float | None = None
    precision: float | None = None
    hallucination_rate: float | None = None
    exact_match: bool = False
