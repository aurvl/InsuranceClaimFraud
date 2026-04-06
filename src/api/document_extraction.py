from __future__ import annotations

import json
import re
from io import BytesIO
from typing import Any

from PIL import Image
from pypdf import PdfReader


CANONICAL_FIELDS = [
    "claim_id",
    "customer_id",
    "claim_date",
    "policy_start_date",
    "policy_age_days",
    "claim_amount",
    "claim_type",
    "customer_age",
    "num_previous_claims",
    "time_since_last_claim_days",
    "service_provider_id",
    "location",
    "weather_condition",
]

OPTIONAL_FIELDS = {
    "time_since_last_claim_days",
}

JSON_FIELD_ALIASES = {
    "claim_id": "claim_id",
    "customer_id": "customer_id",
    "claim_date": "claim_date",
    "policy_start_date": "policy_start_date",
    "policy_age_days": "policy_age_days",
    "claim_amount": "claim_amount",
    "claim_type": "claim_type",
    "customer_age": "customer_age",
    "num_previous_claims": "num_previous_claims",
    "previous_claims": "num_previous_claims",
    "time_since_last_claim_days": "time_since_last_claim_days",
    "service_provider_id": "service_provider_id",
    "garage_id": "service_provider_id",
    "location": "location",
    "weather_condition": "weather_condition",
}

PDF_LINE_MAP = {
    "Claim ID:": "claim_id",
    "Customer ID:": "customer_id",
    "Claim Date:": "claim_date",
    "Policy Start Date:": "policy_start_date",
    "Policy Age (days):": "policy_age_days",
    "Claim Type:": "claim_type",
    "Claim Amount:": "claim_amount",
    "Customer Age:": "customer_age",
    "Previous Claims:": "num_previous_claims",
    "Time Since Last Claim:": "time_since_last_claim_days",
    "Service Provider:": "service_provider_id",
    "Location:": "location",
    "Weather:": "weather_condition",
}


def normalize_value(field_name: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    if field_name in {"policy_age_days", "customer_age", "num_previous_claims"}:
        return int(value)
    if field_name == "claim_amount":
        if isinstance(value, str):
            cleaned = value.replace("EUR", "").replace(",", "").strip()
            return float(cleaned)
        return float(value)
    if field_name == "time_since_last_claim_days":
        if value in {"N/A", "n/a", "", None}:
            return None
        if isinstance(value, str):
            matched = re.search(r"(\d+)", value)
            return int(matched.group(1)) if matched else None
        return int(value)
    if field_name in {"claim_date", "policy_start_date"}:
        return str(value)
    return str(value)


def parse_claim_json_bytes(claim_json_bytes: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(claim_json_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError("claim_json must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("claim_json must decode to a JSON object.")
    extracted: dict[str, Any] = {}
    for field_name, value in payload.items():
        canonical_name = JSON_FIELD_ALIASES.get(field_name)
        if canonical_name is None:
            continue
        extracted[canonical_name] = normalize_value(canonical_name, value)
    return extracted


def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    text_chunks: list[str] = []
    for page in reader.pages:
        text_chunks.append(page.extract_text() or "")
    return "\n".join(text_chunks)


def extract_fields_from_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    text = extract_pdf_text(pdf_bytes)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    extracted: dict[str, Any] = {}
    for prefix, field_name in PDF_LINE_MAP.items():
        for line in lines:
            if line.startswith(prefix):
                raw_value = line.replace(prefix, "", 1).strip()
                extracted[field_name] = normalize_value(field_name, raw_value)
                break
    extracted["_pdf_text"] = text
    return extracted


def extract_fields_from_image(filename: str, image_bytes: bytes) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    claim_match = re.search(r"(CLM-\d{5}-\d{3})", filename, flags=re.IGNORECASE)
    customer_match = re.search(r"(CUST-\d{5})", filename, flags=re.IGNORECASE)
    if claim_match:
        extracted["claim_id"] = claim_match.group(1).upper()
    if customer_match:
        extracted["customer_id"] = customer_match.group(1).upper()

    with Image.open(BytesIO(image_bytes)) as image:
        extracted["_image_metadata"] = {
            "width": int(image.width),
            "height": int(image.height),
            "format": image.format,
        }
    return extracted


def build_source_map(
    json_fields: dict[str, Any],
    pdf_fields: dict[str, Any],
    image_fields: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    source_map: dict[str, list[dict[str, Any]]] = {}
    for source_name, source_fields in (
        ("json", json_fields),
        ("pdf", pdf_fields),
        ("image", image_fields),
    ):
        for field_name, value in source_fields.items():
            if field_name.startswith("_"):
                continue
            if value is None:
                continue
            source_map.setdefault(field_name, []).append({"source": source_name, "value": value})
    return source_map


def detect_mismatches(source_map: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for field_name, entries in source_map.items():
        distinct_values = {json.dumps(entry["value"], sort_keys=True, default=str) for entry in entries}
        if len(distinct_values) > 1:
            mismatches.append(
                {
                    "field": field_name,
                    "values": entries,
                }
            )
    return mismatches


def consolidate_claim_fields(
    json_fields: dict[str, Any],
    pdf_fields: dict[str, Any],
    image_fields: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    consolidated: dict[str, Any] = {}
    chosen_sources: dict[str, str] = {}
    missing_fields: list[str] = []

    for field_name in CANONICAL_FIELDS:
        for source_name, source_fields in (
            ("json", json_fields),
            ("pdf", pdf_fields),
            ("image", image_fields),
        ):
            if field_name in source_fields and source_fields[field_name] is not None:
                consolidated[field_name] = source_fields[field_name]
                chosen_sources[field_name] = source_name
                break
        else:
            if field_name not in OPTIONAL_FIELDS:
                missing_fields.append(field_name)
            else:
                consolidated[field_name] = None

    return consolidated, chosen_sources, missing_fields


def extract_and_consolidate_claim_package(
    *,
    claim_json_bytes: bytes,
    pdf_bytes: bytes | None,
    image_filename: str | None,
    image_bytes: bytes | None,
) -> dict[str, Any]:
    json_fields = parse_claim_json_bytes(claim_json_bytes)
    pdf_fields = extract_fields_from_pdf(pdf_bytes) if pdf_bytes else {}
    image_fields = extract_fields_from_image(image_filename or "image", image_bytes) if image_bytes else {}

    source_map = build_source_map(
        json_fields=json_fields,
        pdf_fields=pdf_fields,
        image_fields=image_fields,
    )
    mismatches = detect_mismatches(source_map=source_map)
    consolidated_claim, chosen_sources, missing_fields = consolidate_claim_fields(
        json_fields=json_fields,
        pdf_fields=pdf_fields,
        image_fields=image_fields,
    )

    return {
        "json_fields": json_fields,
        "pdf_fields": {key: value for key, value in pdf_fields.items() if not key.startswith("_")},
        "image_fields": {key: value for key, value in image_fields.items() if not key.startswith("_")},
        "pdf_text_excerpt": pdf_fields.get("_pdf_text", "")[:1200],
        "image_metadata": image_fields.get("_image_metadata", {}),
        "source_map": source_map,
        "chosen_sources": chosen_sources,
        "mismatches": mismatches,
        "missing_fields": missing_fields,
        "structured_claim": consolidated_claim,
        "document_extraction_mode": "implemented_rule_based_pdf_and_image_extraction",
    }
