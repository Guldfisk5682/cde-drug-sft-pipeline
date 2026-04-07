from __future__ import annotations

import hashlib
import re
from typing import Iterable

from leaflet_pipeline.leaflet_cleaning import clean_section_text
from leaflet_pipeline.leaflet_schema import LeafletExtraction
from leaflet_pipeline.pipeline_schema import ChatMessage, CleanLeafletRecord, QASample

REQUIRED_FIELDS: tuple[str, ...] = (
    "otc_or_rx_flag",
    "indications",
    "dosage_and_administration",
    "contraindications",
)

DRUG_APPROVAL_PREFIX_PATTERN = re.compile(
    r"^(?:国?药准字|国药准字号|药准字)[A-Za-z0-9]+(?:[_-]+)?"
)
PAGE_NUMBER_PATTERN = re.compile(r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页")
BROKEN_LINE_PATTERN = re.compile(r"(?<=[\u4e00-\u9fffA-Za-z0-9])\s*\n\s*(?=[\u4e00-\u9fffA-Za-z0-9])")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；])")
DOSAGE_UNIT_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?)\s*(?:mg|g|ml|mL|μg|ug)\b|(?:[一二三四五六七八九十\d]+)\s*(?:片|粒|袋|丸|支|滴|喷|揿)"
)
DOSAGE_AMOUNT_CUE_PATTERN = re.compile(r"(?:每次|一次|单次|起始剂量|推荐剂量|最大剂量)")
DOSAGE_FREQUENCY_PATTERN = re.compile(r"(?:一日|每日|每天|每晚|每晨|每周|每\d+小时|次/日|次／日|次/天|次／天)")
POPULATION_PATTERN = re.compile(r"(?:儿童|老年|孕妇|哺乳期|肝功能|肾功能|透析|特殊人群)")

FIELD_TEMPLATE_POOLS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "otc_or_rx_flag": {
        "train": (
            ("rx_train_1", "{drug_name}是处方药还是非处方药？"),
            ("rx_train_2", "{drug_name}属于处方药还是非处方药？"),
            ("rx_train_3", "按药品类别来看，{drug_name}是处方药还是非处方药？"),
        ),
        "test": (
            ("rx_test_1", "{drug_name}需要凭处方使用，还是属于非处方药？"),
        ),
    },
    "indications": {
        "train": (
            ("ind_train_1", "{drug_name}的适应症是什么？"),
            ("ind_train_2", "{drug_name}主要用于治疗什么？"),
            ("ind_train_3", "{drug_name}通常适用于哪些情况？"),
        ),
        "test": (
            ("ind_test_1", "{drug_name}一般用于哪些疾病或症状？"),
        ),
    },
    "contraindications": {
        "train": (
            ("con_train_1", "{drug_name}的禁忌或警告是什么？"),
            ("con_train_2", "{drug_name}有哪些禁忌事项？"),
            ("con_train_3", "{drug_name}在禁忌或警告方面需要注意什么？"),
        ),
        "test": (
            ("con_test_1", "{drug_name}有哪些人群或情况应避免使用？"),
        ),
    },
}

DOSAGE_TEMPLATE_POOLS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "dosage_amount": {
        "train": (
            ("dose_amt_train_1", "{drug_name}一次用量通常是多少？"),
            ("dose_amt_train_2", "{drug_name}每次应该用多少？"),
        ),
        "test": (
            ("dose_amt_test_1", "{drug_name}单次服用剂量一般是多少？"),
        ),
    },
    "dosage_frequency": {
        "train": (
            ("dose_freq_train_1", "{drug_name}一天用几次？"),
            ("dose_freq_train_2", "{drug_name}每日或一日的用药频次是什么？"),
            ("dose_freq_train_3", "{drug_name}通常多久用一次？"),
        ),
        "test": (
            ("dose_freq_test_1", "{drug_name}常规的用药频次怎么写？"),
        ),
    },
    "dosage_population": {
        "train": (
            ("dose_pop_train_1", "{drug_name}针对特殊人群的用法用量说明是什么？"),
            ("dose_pop_train_2", "{drug_name}在特殊人群中的用法用量要点是什么？"),
            ("dose_pop_train_3", "{drug_name}若涉及儿童、老年人等人群，用法用量如何说明？"),
        ),
        "test": (
            ("dose_pop_test_1", "{drug_name}若涉及特殊人群，用法用量应如何说明？"),
        ),
    },
    "dosage_general": {
        "train": (
            ("dose_gen_train_1", "{drug_name}的用法用量是什么？"),
            ("dose_gen_train_2", "请说明{drug_name}的具体用法用量。"),
            ("dose_gen_train_3", "{drug_name}应该怎么用、用多少？"),
        ),
        "test": (
            ("dose_gen_test_1", "{drug_name}应如何使用，剂量和频次怎么写？"),
        ),
    },
}

RX_OTC_ANSWER_POOLS: dict[str, tuple[tuple[str, str], ...]] = {
    "RX": (
        ("rx_ans_1", "{drug_name}是处方药，请在医师指导下使用。"),
        ("rx_ans_2", "该药属于处方药，需在医师指导下使用。"),
        ("rx_ans_3", "这是处方药，应遵医嘱使用。"),
    ),
    "OTC": (
        ("otc_ans_1", "{drug_name}是非处方药。"),
        ("otc_ans_2", "该药属于非处方药，可按说明书使用。"),
        ("otc_ans_3", "这是非处方药，可在药师指导下使用。"),
    ),
    "UNKNOWN": (
        ("unknown_ans_1", "该药的处方属性暂不明确。"),
    ),
}


def compact_text(text: str | None) -> str | None:
    if not text:
        return None
    text = clean_section_text(text)
    text = PAGE_NUMBER_PATTERN.sub(" ", text)
    text = BROKEN_LINE_PATTERN.sub("", text)
    text = re.sub(r"[-_]{3,}", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip() or None


def normalize_drug_name(drug_name: str | None) -> str | None:
    normalized = compact_text(drug_name)
    if not normalized:
        return None
    normalized = DRUG_APPROVAL_PREFIX_PATTERN.sub("", normalized)
    normalized = re.sub(r"^[\s_\-:：]+", "", normalized)
    return normalized.strip() or None


def normalize_otc_or_rx_answer(flag: str) -> str:
    if flag == "RX":
        return "处方药"
    if flag == "OTC":
        return "非处方药"
    return "未知"


def build_rx_otc_answer(drug_name: str, flag: str, key: str) -> str:
    variants = RX_OTC_ANSWER_POOLS.get(flag, RX_OTC_ANSWER_POOLS["UNKNOWN"])
    variant_id, template = _stable_template_pick(key, variants, 1)[0]
    _ = variant_id
    return template.format(drug_name=drug_name)


def build_clean_record(extraction: LeafletExtraction) -> CleanLeafletRecord:
    record = CleanLeafletRecord(
        doc_id=extraction.doc_id,
        source_pdf_path=extraction.pdf_path,
        drug_name=normalize_drug_name(extraction.drug_name_guess),
        otc_or_rx_flag=extraction.otc_or_rx_flag,
        indications=compact_text(extraction.indications),
        dosage_and_administration=compact_text(extraction.dosage_and_administration),
        contraindications=compact_text(extraction.contraindications),
        warnings=list(extraction.warnings),
    )

    missing_fields: list[str] = []
    if record.otc_or_rx_flag == "UNKNOWN":
        missing_fields.append("otc_or_rx_flag")
    for field_name in ("indications", "dosage_and_administration", "contraindications"):
        value = getattr(record, field_name)
        if not value:
            missing_fields.append(field_name)

    score = 0.0
    score += 0.25 if record.otc_or_rx_flag != "UNKNOWN" else 0.0
    score += 0.25 if record.indications else 0.0
    score += 0.30 if record.dosage_and_administration else 0.0
    score += 0.20 if record.contraindications else 0.0
    if record.dosage_and_administration and len(record.dosage_and_administration) >= 80:
        score += 0.05
    if record.indications and len(record.indications) >= 20:
        score += 0.03
    if record.contraindications and len(record.contraindications) >= 20:
        score += 0.02
    score -= min(0.1, 0.02 * len(record.warnings))

    record.missing_fields = missing_fields
    record.quality_score = round(max(0.0, min(score, 1.0)), 3)
    record.is_valid = not missing_fields
    return record


def _build_sample_id(
    doc_id: str,
    task_type: str,
    split_hint: str,
    template_id: str,
    chunk_index: int | None,
) -> str:
    raw = f"{doc_id}:{task_type}:{split_hint}:{template_id}:{chunk_index}".encode("utf-8")
    digest = hashlib.md5(raw).hexdigest()[:12]
    return f"{doc_id}-{task_type}-{split_hint}-{digest}"


def _stable_template_pick(
    key: str, templates: tuple[tuple[str, str], ...], count: int
) -> list[tuple[str, str]]:
    ranked = sorted(
        templates,
        key=lambda item: hashlib.md5(f"{key}:{item[0]}".encode("utf-8")).hexdigest(),
    )
    return ranked[: min(count, len(ranked))]


def _iter_prompt_variants(
    *,
    doc_id: str,
    field_name: str,
    template_pool: dict[str, tuple[tuple[str, str], ...]],
    train_count: int = 2,
) -> Iterable[tuple[str, str, str]]:
    key = f"{doc_id}:{field_name}"
    for template_id, template in _stable_template_pick(key, template_pool["train"], train_count):
        yield "train", template_id, template
    for template_id, template in _stable_template_pick(key, template_pool["test"], 1):
        yield "test", template_id, template


def split_text_for_qa(
    text: str | None,
    *,
    sentence_window: int = 2,
    min_chunk_chars: int = 12,
    max_chunk_chars: int = 160,
) -> list[str]:
    if not text:
        return []

    normalized = compact_text(text)
    if not normalized:
        return []

    sentences = [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(normalized) if item.strip()]
    if not sentences:
        return [normalized]

    chunks: list[str] = []
    buffer: list[str] = []
    for sentence in sentences:
        buffer.append(sentence)
        joined = "".join(buffer)
        if len(buffer) >= sentence_window or len(joined) >= max_chunk_chars:
            chunks.append(joined)
            buffer = []
    if buffer:
        chunks.append("".join(buffer))

    finalized: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_chars:
            finalized.append(chunk)
            continue
        comma_parts = [part.strip() for part in re.split(r"(?<=[，、])", chunk) if part.strip()]
        temp = ""
        for part in comma_parts:
            if len(temp) + len(part) > max_chunk_chars and temp:
                finalized.append(temp)
                temp = part
            else:
                temp += part
        if temp:
            finalized.append(temp)

    seen: set[str] = set()
    deduped: list[str] = []
    for chunk in finalized:
        chunk = chunk.strip()
        if len(chunk) < min_chunk_chars or chunk in seen:
            continue
        seen.add(chunk)
        deduped.append(chunk)
    return deduped or [normalized]


def classify_dosage_chunk(text: str) -> str:
    if DOSAGE_AMOUNT_CUE_PATTERN.search(text) or DOSAGE_UNIT_PATTERN.search(text):
        return "dosage_amount"
    if POPULATION_PATTERN.search(text):
        return "dosage_population"
    if DOSAGE_FREQUENCY_PATTERN.search(text):
        return "dosage_frequency"
    return "dosage_general"


def _build_field_samples(
    *,
    doc_id: str,
    drug_name: str,
    task_type: str,
    answer_field: str,
    answer: str,
    template_pool: dict[str, tuple[tuple[str, str], ...]],
    chunk_index: int | None = None,
    train_count: int = 2,
    answer_variant_flag: str | None = None,
) -> list[QASample]:
    samples: list[QASample] = []
    for split_hint, template_id, template in _iter_prompt_variants(
        doc_id=doc_id,
        field_name=f"{task_type}:{chunk_index}",
        template_pool=template_pool,
        train_count=train_count,
    ):
        answer_text = answer
        if answer_variant_flag is not None:
            answer_text = build_rx_otc_answer(
                drug_name,
                answer_variant_flag,
                f"{doc_id}:{task_type}:{split_hint}:{template_id}:{chunk_index}",
            )
        samples.append(
            QASample(
                sample_id=_build_sample_id(doc_id, task_type, split_hint, template_id, chunk_index),
                doc_id=doc_id,
                drug_name=drug_name,
                task_type=task_type,
                answer_field=answer_field,
                template_id=template_id,
                split_hint=split_hint,
                chunk_index=chunk_index,
                messages=[
                    ChatMessage(role="user", content=template.format(drug_name=drug_name)),
                    ChatMessage(role="assistant", content=answer_text),
                ],
            )
        )
    return samples


def build_qa_samples(record: CleanLeafletRecord) -> list[QASample]:
    if not record.is_valid or not record.drug_name:
        return []

    samples: list[QASample] = []
    samples.extend(
        _build_field_samples(
            doc_id=record.doc_id,
            drug_name=record.drug_name,
            task_type="rx_otc_qa",
            answer_field="otc_or_rx_flag",
            answer=normalize_otc_or_rx_answer(record.otc_or_rx_flag),
            template_pool=FIELD_TEMPLATE_POOLS["otc_or_rx_flag"],
            train_count=1,
            answer_variant_flag=record.otc_or_rx_flag,
        )
    )

    if record.indications:
        samples.extend(
            _build_field_samples(
                doc_id=record.doc_id,
                drug_name=record.drug_name,
                task_type="indications_qa",
                answer_field="indications",
                answer=record.indications,
                template_pool=FIELD_TEMPLATE_POOLS["indications"],
            )
        )

    if record.contraindications:
        samples.extend(
            _build_field_samples(
                doc_id=record.doc_id,
                drug_name=record.drug_name,
                task_type="contraindications_qa",
                answer_field="contraindications",
                answer=record.contraindications,
                template_pool=FIELD_TEMPLATE_POOLS["contraindications"],
            )
        )

    if record.dosage_and_administration:
        for chunk_index, chunk in enumerate(split_text_for_qa(record.dosage_and_administration)):
            dosage_task_type = classify_dosage_chunk(chunk)
            samples.extend(
                _build_field_samples(
                    doc_id=record.doc_id,
                    drug_name=record.drug_name,
                    task_type=dosage_task_type,
                    answer_field="dosage_and_administration",
                    answer=chunk,
                    template_pool=DOSAGE_TEMPLATE_POOLS[dosage_task_type],
                    chunk_index=chunk_index,
                )
            )
    return samples


def assign_split(doc_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"
