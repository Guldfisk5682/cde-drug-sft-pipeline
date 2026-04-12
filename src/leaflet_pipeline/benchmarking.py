from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from leaflet_pipeline.benchmark_schema import BenchmarkPrediction, BenchmarkSample, BenchmarkSampleResult
from leaflet_pipeline.pipeline_schema import QASample

RX_KEYWORDS = ("处方药",)
OTC_KEYWORDS = ("非处方药",)

ROUTE_KEYWORDS = (
    "口服",
    "外用",
    "含服",
    "舌下含服",
    "静脉注射",
    "静脉滴注",
    "肌内注射",
    "皮下注射",
    "滴眼",
    "滴鼻",
    "吸入",
    "冲服",
    "涂抹",
    "贴敷",
)

POPULATION_ALIASES = {
    "儿童": "儿童",
    "婴儿": "儿童",
    "新生儿": "儿童",
    "小儿": "儿童",
    "老年人": "老年人",
    "老年患者": "老年人",
    "孕妇": "孕妇",
    "妊娠期妇女": "孕妇",
    "哺乳期妇女": "哺乳期",
    "哺乳期": "哺乳期",
    "肝功能不全": "肝功能不全",
    "肾功能不全": "肾功能不全",
    "透析患者": "透析患者",
    "成人": "成人",
}

STOPWORD_PREFIX_PATTERN = re.compile(
    r"^(?:本品|该药|用于|适用于|主要用于|可用于|一般用于|应避免|避免|禁用于|禁忌|警告|慎用|请勿|不宜|不得|应在|需在)+"
)
CLAUSE_SPLIT_PATTERN = re.compile(r"[。；;！!？?\n]+")
PHRASE_SPLIT_PATTERN = re.compile(r"[，、,：:（）()]+")

DOSE_PATTERNS = (
    re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|ml|mL|μg|ug|IU|万IU|U)", re.IGNORECASE),
    re.compile(r"(?:半|[一二三四五六七八九十两\d]+)\s*(?:片|粒|袋|丸|支|滴|喷|揿)"),
)

FREQUENCY_PATTERNS = (
    re.compile(r"(?:一日|每日|每天|每晚|每晨)\s*(?:\d+|[一二三四五六七八九十两])\s*次"),
    re.compile(r"(?:每\d+\s*小时\s*1次)"),
    re.compile(r"(?:每周\s*\d+\s*次)"),
    re.compile(r"(?:bid|tid|qid|q\d+h)", re.IGNORECASE),
)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("：", ":")
    return text.strip()


def normalized_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = text.replace(" ", "")
    return text


def normalize_phrase(text: str) -> str:
    text = normalize_text(text)
    text = STOPWORD_PREFIX_PATTERN.sub("", text)
    text = text.strip(" ：:，、。；;（）()[]【】")
    return text


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        key = normalized_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def extract_rx_label(text: str) -> str:
    normalized = normalize_text(text)
    has_rx = any(keyword in normalized for keyword in RX_KEYWORDS)
    has_otc = any(keyword in normalized for keyword in OTC_KEYWORDS)
    if has_rx and not has_otc:
        return "RX"
    if has_otc and not has_rx:
        return "OTC"
    return "UNKNOWN"


def extract_dose_values(text: str) -> list[str]:
    values: list[str] = []
    for pattern in DOSE_PATTERNS:
        values.extend(match.group(0) for match in pattern.finditer(text))
    return unique_preserve_order(values)


def extract_frequency_values(text: str) -> list[str]:
    values: list[str] = []
    for pattern in FREQUENCY_PATTERNS:
        values.extend(match.group(0) for match in pattern.finditer(text))
    return unique_preserve_order(values)


def extract_route_values(text: str) -> list[str]:
    return [keyword for keyword in ROUTE_KEYWORDS if keyword in text]


def extract_population_values(text: str) -> list[str]:
    matched: list[str] = []
    for raw, normalized in POPULATION_ALIASES.items():
        if raw in text:
            matched.append(normalized)
    return unique_preserve_order(matched)


def extract_clause_keywords(text: str, *, max_keywords: int = 8) -> list[str]:
    normalized = normalize_text(text)
    clauses: list[str] = []
    for clause in CLAUSE_SPLIT_PATTERN.split(normalized):
        clause = normalize_phrase(clause)
        if not clause:
            continue
        phrases = [normalize_phrase(part) for part in PHRASE_SPLIT_PATTERN.split(clause)]
        for phrase in phrases:
            if 2 <= len(phrase) <= 24:
                clauses.append(phrase)
    return unique_preserve_order(clauses)[:max_keywords]


def extract_slots(task_type: str, answer: str) -> dict[str, list[str]]:
    text = normalize_text(answer)
    if not text:
        return {}
    if task_type == "rx_otc_qa":
        return {}

    if task_type == "dosage_amount":
        return {"dose": extract_dose_values(text)}
    if task_type == "dosage_frequency":
        return {"frequency": extract_frequency_values(text)}
    if task_type == "dosage_population":
        return {"population": extract_population_values(text)}

    if task_type == "dosage_general":
        return {
            "dose": extract_dose_values(text),
            "frequency": extract_frequency_values(text),
            "route": extract_route_values(text),
            "population": extract_population_values(text),
            "keywords": extract_clause_keywords(text, max_keywords=6),
        }

    if task_type in {"indications_qa", "contraindications_qa"}:
        return {
            "population": extract_population_values(text),
            "keywords": extract_clause_keywords(text, max_keywords=8),
        }

    return {"keywords": extract_clause_keywords(text, max_keywords=8)}


def flatten_slots(slots: dict[str, list[str]]) -> set[str]:
    flattened: set[str] = set()
    for values in slots.values():
        for value in values:
            key = normalized_key(value)
            if key:
                flattened.add(key)
    return flattened


def iter_qa_samples(path: Path) -> list[QASample]:
    samples: list[QASample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(QASample.model_validate_json(line))
    return samples


def build_benchmark_sample(sample: QASample) -> BenchmarkSample:
    question = sample.messages[0].content
    reference_answer = sample.messages[1].content
    benchmark_id = hashlib.md5(f"{sample.sample_id}:benchmark".encode("utf-8")).hexdigest()[:12]
    answer_type = "classification" if sample.task_type == "rx_otc_qa" else "span"
    gold_label = extract_rx_label(reference_answer) if answer_type == "classification" else None
    gold_slots = {} if answer_type == "classification" else extract_slots(sample.task_type, reference_answer)
    return BenchmarkSample(
        benchmark_id=f"{sample.doc_id}-{benchmark_id}",
        source_sample_id=sample.sample_id,
        doc_id=sample.doc_id,
        drug_name=sample.drug_name,
        task_type=sample.task_type,
        question=question,
        reference_answer=reference_answer,
        answer_type=answer_type,
        gold_label=gold_label,
        gold_slots=gold_slots,
        template_id=sample.template_id,
        chunk_index=sample.chunk_index,
    )


def evaluate_prediction(sample: BenchmarkSample, prediction_text: str) -> BenchmarkSampleResult:
    normalized_reference = normalize_text(sample.reference_answer)
    normalized_prediction = normalize_text(prediction_text)
    exact_match = normalized_key(normalized_reference) == normalized_key(normalized_prediction)

    if sample.answer_type == "classification":
        predicted_label = extract_rx_label(prediction_text)
        return BenchmarkSampleResult(
            benchmark_id=sample.benchmark_id,
            task_type=sample.task_type,
            answer_type=sample.answer_type,
            question=sample.question,
            reference_answer=sample.reference_answer,
            prediction=prediction_text,
            gold_label=sample.gold_label,
            predicted_label=predicted_label,
            label_correct=predicted_label == sample.gold_label,
            exact_match=exact_match,
        )

    predicted_slots = extract_slots(sample.task_type, prediction_text)
    gold_flat = flatten_slots(sample.gold_slots)
    predicted_flat = flatten_slots(predicted_slots)
    matched_flat = gold_flat & predicted_flat
    missing_flat = gold_flat - predicted_flat
    hallucinated_flat = predicted_flat - gold_flat

    recall = len(matched_flat) / len(gold_flat) if gold_flat else 1.0
    precision = len(matched_flat) / len(predicted_flat) if predicted_flat else 0.0
    hallucination_rate = len(hallucinated_flat) / len(predicted_flat) if predicted_flat else 0.0

    return BenchmarkSampleResult(
        benchmark_id=sample.benchmark_id,
        task_type=sample.task_type,
        answer_type=sample.answer_type,
        question=sample.question,
        reference_answer=sample.reference_answer,
        prediction=prediction_text,
        gold_slots=sample.gold_slots,
        predicted_slots=predicted_slots,
        matched_values=sorted(matched_flat),
        missing_values=sorted(missing_flat),
        hallucinated_values=sorted(hallucinated_flat),
        recall=round(recall, 4),
        precision=round(precision, 4),
        hallucination_rate=round(hallucination_rate, 4),
        exact_match=exact_match,
    )


def summarize_results(results: list[BenchmarkSampleResult]) -> dict[str, object]:
    def summarize_bucket(bucket: list[BenchmarkSampleResult]) -> dict[str, float | int]:
        total = len(bucket)
        if total == 0:
            return {"count": 0}
        cls_bucket = [item for item in bucket if item.answer_type == "classification"]
        span_bucket = [item for item in bucket if item.answer_type == "span"]

        summary: dict[str, float | int] = {"count": total}
        if cls_bucket:
            correct = sum(1 for item in cls_bucket if item.label_correct)
            summary["classification_accuracy"] = round(correct / len(cls_bucket), 4)
            summary["classification_count"] = len(cls_bucket)
        if span_bucket:
            summary["span_count"] = len(span_bucket)
            summary["avg_recall"] = round(
                sum(item.recall or 0.0 for item in span_bucket) / len(span_bucket), 4
            )
            summary["avg_precision"] = round(
                sum(item.precision or 0.0 for item in span_bucket) / len(span_bucket), 4
            )
            summary["avg_hallucination_rate"] = round(
                sum(item.hallucination_rate or 0.0 for item in span_bucket) / len(span_bucket), 4
            )
            summary["exact_match_rate"] = round(
                sum(1 for item in span_bucket if item.exact_match) / len(span_bucket), 4
            )
        return summary

    by_task: dict[str, list[BenchmarkSampleResult]] = defaultdict(list)
    for result in results:
        by_task[result.task_type].append(result)

    return {
        "overall": summarize_bucket(results),
        "by_task_type": {task: summarize_bucket(bucket) for task, bucket in sorted(by_task.items())},
    }


def iter_benchmark_samples(path: Path) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(BenchmarkSample.model_validate_json(line))
    return samples


def iter_benchmark_predictions(path: Path) -> list[BenchmarkPrediction]:
    predictions: list[BenchmarkPrediction] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            predictions.append(BenchmarkPrediction.model_validate_json(line))
    return predictions


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            if hasattr(row, "model_dump"):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
