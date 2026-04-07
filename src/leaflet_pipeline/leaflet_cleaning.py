from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from leaflet_pipeline.leaflet_schema import LeafletExtraction, LeafletSection

SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "indications": (
        "适应症",
        "功能主治",
    ),
    "dosage_and_administration": (
        "用法用量",
        "用法与用量",
    ),
    "contraindications": (
        "禁忌",
        "警告",
    ),
}

BOUNDARY_HEADINGS: tuple[str, ...] = (
    "成份",
    "性状",
    "适应症",
    "功能主治",
    "规格",
    "用法用量",
    "用法与用量",
    "不良反应",
    "不良事件",
    "禁忌",
    "警告",
    "注意事项",
    "孕妇及哺乳期妇女用药",
    "儿童用药",
    "老年用药",
    "药物相互作用",
    "药理毒理",
    "药代动力学",
    "贮藏",
    "包装",
    "有效期",
    "执行标准",
)


def normalize_heading_text(text: str) -> str:
    text = text.strip()
    text = text.replace("【", "").replace("】", "")
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"[\s:：·•]+", "", text)
    return text


BOUNDARY_HEADINGS_NORMALIZED = {normalize_heading_text(item) for item in BOUNDARY_HEADINGS}


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    pages: list[str] = []
    warnings: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            if not text.strip():
                warnings.append(f"empty_page:{page_number}")
            pages.append(text)
    return "\n\n".join(pages), warnings


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def guess_drug_name(pdf_path: Path, lines: list[str]) -> str | None:
    stem = pdf_path.stem
    stem = re.sub(r"^\d+_[0-9a-f]{10}_", "", stem)
    stem = stem.replace("_", "")
    stem = re.sub(r"\.pdf$", "", stem, flags=re.IGNORECASE)
    if stem:
        return stem
    for line in lines[:10]:
        if 2 <= len(line) <= 40 and any(token in line for token in ("片", "胶囊", "颗粒", "散", "丸", "口服液")):
            return line
    return None


def detect_otc_or_rx(text: str) -> tuple[str, list[str]]:
    lowered = text.lower()
    evidence: list[str] = []
    if "请仔细阅读说明书并在医师指导下使用" in text or "处方药" in text:
        evidence.append("rx_hint")
        return "RX", evidence
    if "otc" in lowered or "非处方药" in text or "甲类非处方药" in text or "乙类非处方药" in text:
        evidence.append("otc_hint")
        return "OTC", evidence
    return "UNKNOWN", evidence


def parse_heading_line(line: str) -> tuple[str | None, str]:
    line = line.strip()
    match = re.match(r"^【\s*(.*?)\s*】\s*(.*)$", line)
    if match:
        return normalize_heading_text(match.group(1)), match.group(2).strip()

    match = re.match(r"^([^\s:：]{2,})\s*[:：]\s*(.*)$", line)
    if match:
        return normalize_heading_text(match.group(1)), match.group(2).strip()

    normalized = normalize_heading_text(line)
    if normalized in BOUNDARY_HEADINGS_NORMALIZED:
        return normalized, ""
    return None, ""


def _find_section_start(lines: list[str], aliases: tuple[str, ...]) -> tuple[int | None, str | None, str]:
    alias_map = {normalize_heading_text(alias): alias for alias in aliases}
    for idx, line in enumerate(lines):
        heading_normalized, remainder = parse_heading_line(line)
        if heading_normalized in alias_map:
            return idx, alias_map[heading_normalized], remainder
    return None, None, ""


def _find_section_end(lines: list[str], start_idx: int) -> int:
    for idx in range(start_idx + 1, len(lines)):
        heading_normalized, _ = parse_heading_line(lines[idx])
        if heading_normalized in BOUNDARY_HEADINGS_NORMALIZED:
            return idx
    return len(lines)


def extract_sections(lines: list[str]) -> dict[str, LeafletSection]:
    sections: dict[str, LeafletSection] = {}
    for field_name, aliases in SECTION_PATTERNS.items():
        start_idx, matched_alias, remainder = _find_section_start(lines, aliases)
        if start_idx is None or matched_alias is None:
            continue
        end_idx = _find_section_end(lines, start_idx)
        section_lines = lines[start_idx + 1:end_idx]
        if remainder:
            section_lines = [remainder, *section_lines]
        if not section_lines:
            continue
        sections[field_name] = LeafletSection(
            heading=matched_alias,
            text="\n".join(section_lines).strip(),
            start_line=start_idx + 1,
            end_line=end_idx,
        )
    return sections


def clean_section_text(text: str) -> str:
    text = re.sub(r"^\s*[【\[]?.+?[】\]]?\s*", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"(?<=\d)\s+(?=(mg|g|ml|mL|片|粒|袋|次|日))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\D)\s+(?=\d+(mg|g|ml|mL))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"一\s*日\s*", "一日", text)
    text = re.sub(r"每\s*日\s*", "每日", text)
    text = re.sub(r"一\s*次\s*", "一次", text)
    text = re.sub(r"\s*[:：]\s*", "：", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_leaflet_fields(pdf_path: Path) -> LeafletExtraction:
    raw_text, warnings = extract_text_from_pdf(pdf_path)
    normalized = normalize_text(raw_text)
    lines = split_lines(normalized)
    sections = extract_sections(lines)
    otc_or_rx_flag, rx_evidence = detect_otc_or_rx(normalized)
    warnings.extend(rx_evidence)

    extraction = LeafletExtraction(
        doc_id=pdf_path.stem,
        pdf_path=str(pdf_path),
        drug_name_guess=guess_drug_name(pdf_path, lines),
        otc_or_rx_flag=otc_or_rx_flag,
        sections=sections,
        warnings=warnings,
    )

    if "indications" in sections:
        extraction.indications = clean_section_text(sections["indications"].text)
    else:
        extraction.warnings.append("missing_section:indications")

    if "dosage_and_administration" in sections:
        extraction.dosage_and_administration = clean_section_text(
            sections["dosage_and_administration"].text
        )
    else:
        extraction.warnings.append("missing_section:dosage_and_administration")

    if "contraindications" in sections:
        extraction.contraindications = clean_section_text(sections["contraindications"].text)
    else:
        extraction.warnings.append("missing_section:contraindications")

    return extraction
