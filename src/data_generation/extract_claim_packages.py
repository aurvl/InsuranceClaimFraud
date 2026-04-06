from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.api.document_extraction import CANONICAL_FIELDS, extract_and_consolidate_claim_package
from src.config import (
    DATA_SUMMARY_PATH,
    GROUND_TRUTH_PROD_PATH,
    GROUND_TRUTH_TRAIN_PATH,
    PROD_CLIENT_DIR,
    PROD_EXTRACTION_AUDIT_JSON_PATH,
    PROD_IMAGE_DIR,
    PROD_PDF_DIR,
    TRAIN_CLIENT_DIR,
    TRAIN_EXTRACTION_AUDIT_JSON_PATH,
    TRAIN_IMAGE_DIR,
    TRAIN_PDF_DIR,
    ensure_project_directories,
)
from src.db.load_to_postgres import (
    create_supporting_tables,
    rebuild_supporting_tables,
    upsert_historical_claims,
    upsert_production_claim_intakes,
)


STRUCTURED_COLUMNS = CANONICAL_FIELDS + ["is_fraud"]


def load_ground_truth_labels(json_path: Path) -> pd.DataFrame:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dataframe = pd.DataFrame.from_records(payload)
    if dataframe.empty:
        raise ValueError(f"No ground-truth claims found in {json_path}.")
    for column in ("claim_date", "policy_start_date"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column])
    return dataframe.loc[:, STRUCTURED_COLUMNS].copy()


def matching_attachment_path(
    json_path: Path,
    attachment_suffix: str,
    shared_root: Path,
) -> Path | None:
    direct_match = json_path.with_suffix(attachment_suffix)
    if direct_match.exists():
        return direct_match

    shared_match = shared_root / direct_match.name
    if shared_match.exists():
        return shared_match
    return None


def validate_structured_claim(claim_payload: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        claim_date = pd.to_datetime(claim_payload["claim_date"]).date()
        policy_start_date = pd.to_datetime(claim_payload["policy_start_date"]).date()
        computed_policy_age = (claim_date - policy_start_date).days
        if computed_policy_age != int(claim_payload["policy_age_days"]):
            return False, "policy_age_days mismatch"
        return True, None
    except Exception as exc:
        return False, str(exc)


def audit_row_from_result(
    *,
    json_path: Path,
    pdf_path: Path | None,
    image_path: Path | None,
    extraction_result: dict[str, Any],
    validation_error: str | None,
    label_joined: bool,
) -> dict[str, Any]:
    structured_claim = extraction_result.get("structured_claim", {})
    chosen_sources = extraction_result.get("chosen_sources", {})
    return {
        "claim_id": structured_claim.get("claim_id"),
        "customer_id": structured_claim.get("customer_id"),
        "json_path": str(json_path),
        "pdf_path": None if pdf_path is None else str(pdf_path),
        "image_path": None if image_path is None else str(image_path),
        "has_pdf": pdf_path is not None,
        "has_image": image_path is not None,
        "missing_field_count": len(extraction_result.get("missing_fields", [])),
        "mismatch_count": len(extraction_result.get("mismatches", [])),
        "missing_fields": extraction_result.get("missing_fields", []),
        "mismatch_fields": [mismatch["field"] for mismatch in extraction_result.get("mismatches", [])],
        "json_selected_fields": sum(1 for source in chosen_sources.values() if source == "json"),
        "pdf_selected_fields": sum(1 for source in chosen_sources.values() if source == "pdf"),
        "image_selected_fields": sum(1 for source in chosen_sources.values() if source == "image"),
        "extraction_success": not extraction_result.get("missing_fields") and validation_error is None,
        "validation_error": validation_error,
        "label_joined": label_joined,
        "document_extraction_mode": extraction_result.get("document_extraction_mode"),
    }


def write_audit_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def extract_split_packages(
    *,
    split_name: str,
    client_root: Path,
    pdf_root: Path,
    image_root: Path,
        ground_truth_path: Path | None,
    audit_output_path: Path,
) -> dict[str, Any]:
    label_lookup: dict[str, Any] = {}
    if ground_truth_path is not None and ground_truth_path.exists():
        ground_truth_df = load_ground_truth_labels(ground_truth_path)
        label_lookup = ground_truth_df.set_index("claim_id")["is_fraud"].to_dict()

    json_paths = sorted(client_root.rglob("claim_*.json"))
    if not json_paths:
        raise ValueError(f"No client packages found in {client_root}.")

    structured_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for json_path in json_paths:
        pdf_path = matching_attachment_path(json_path=json_path, attachment_suffix=".pdf", shared_root=pdf_root)
        image_path = matching_attachment_path(json_path=json_path, attachment_suffix=".png", shared_root=image_root)
        extraction_result = extract_and_consolidate_claim_package(
            claim_json_bytes=json_path.read_bytes(),
            pdf_bytes=None if pdf_path is None else pdf_path.read_bytes(),
            image_filename=None if image_path is None else image_path.name,
            image_bytes=None if image_path is None else image_path.read_bytes(),
        )
        structured_claim = dict(extraction_result["structured_claim"])
        validation_ok, validation_error = validate_structured_claim(structured_claim)
        claim_id = structured_claim.get("claim_id")
        label_joined = bool(label_lookup) and claim_id in label_lookup
        if label_joined:
            structured_claim["is_fraud"] = int(label_lookup[claim_id])

        audit_rows.append(
            audit_row_from_result(
                json_path=json_path,
                pdf_path=pdf_path,
                image_path=image_path,
                extraction_result=extraction_result,
                validation_error=validation_error,
                label_joined=label_joined,
            )
        )

        if extraction_result["missing_fields"] or not validation_ok:
            continue

        structured_row = {
            "claim_id": structured_claim["claim_id"],
            "customer_id": structured_claim["customer_id"],
            "claim_date": str(structured_claim["claim_date"]),
            "policy_start_date": str(structured_claim["policy_start_date"]),
            "policy_age_days": int(structured_claim["policy_age_days"]),
            "claim_amount": float(structured_claim["claim_amount"]),
            "claim_type": str(structured_claim["claim_type"]),
            "customer_age": int(structured_claim["customer_age"]),
            "num_previous_claims": int(structured_claim["num_previous_claims"]),
            "time_since_last_claim_days": (
                None
                if structured_claim.get("time_since_last_claim_days") is None
                else int(structured_claim["time_since_last_claim_days"])
            ),
            "service_provider_id": str(structured_claim["service_provider_id"]),
            "location": str(structured_claim["location"]),
            "weather_condition": str(structured_claim["weather_condition"]),
            "structured_claim": structured_claim,
            "document_extraction_mode": extraction_result["document_extraction_mode"],
            "chosen_sources": extraction_result.get("chosen_sources", {}),
            "mismatches": extraction_result.get("mismatches", []),
            "package_metadata": {
                "json_path": str(json_path),
                "pdf_path": None if pdf_path is None else str(pdf_path),
                "image_path": None if image_path is None else str(image_path),
            },
        }

        if split_name == "train":
            if not label_joined:
                continue
            structured_row["is_fraud"] = int(structured_claim["is_fraud"])
        structured_rows.append(structured_row)

    audit_payload = {
        "split": split_name,
        "package_count": int(len(json_paths)),
        "structured_rows": int(len(structured_rows)),
        "pdf_coverage": round(sum(item["has_pdf"] for item in audit_rows) / len(audit_rows), 4) if audit_rows else 0.0,
        "image_coverage": round(sum(item["has_image"] for item in audit_rows) / len(audit_rows), 4) if audit_rows else 0.0,
        "successful_extractions": int(sum(item["extraction_success"] for item in audit_rows)),
        "mismatch_packages": int(sum(item["mismatch_count"] > 0 for item in audit_rows)),
        "missing_field_packages": int(sum(item["missing_field_count"] > 0 for item in audit_rows)),
        "audit_rows": audit_rows,
    }
    write_audit_json(audit_output_path, audit_payload)

    if structured_rows:
        structured_df = pd.DataFrame(structured_rows)
        if split_name == "train":
            upsert_historical_claims(
                structured_df[
                    [
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
                        "is_fraud",
                    ]
                ],
            )
        else:
            upsert_production_claim_intakes(
                structured_df[
                    [
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
                        "document_extraction_mode",
                        "chosen_sources",
                        "mismatches",
                        "package_metadata",
                        "structured_claim",
                    ]
                ],
            )

    summary = {
        "split": split_name,
        "package_count": int(len(json_paths)),
        "structured_rows": int(len(structured_rows)),
        "pdf_coverage": audit_payload["pdf_coverage"],
        "image_coverage": audit_payload["image_coverage"],
        "successful_extractions": audit_payload["successful_extractions"],
        "mismatch_packages": audit_payload["mismatch_packages"],
        "missing_field_packages": audit_payload["missing_field_packages"],
        "audit_output_path": str(audit_output_path),
        "ground_truth_path": None if ground_truth_path is None else str(ground_truth_path),
        "postgres_target": "claims" if split_name == "train" else "production_claim_intake",
        "structured_intermediate_written": "none",
    }
    return summary


def main(split: str = "all", reset_db: bool = False) -> dict[str, Any]:
    ensure_project_directories()
    if reset_db:
        rebuild_supporting_tables()
    else:
        create_supporting_tables()
    summaries: dict[str, Any] = {}

    if split in {"all", "train"}:
        summaries["train"] = extract_split_packages(
            split_name="train",
            client_root=TRAIN_CLIENT_DIR,
            pdf_root=TRAIN_PDF_DIR,
            image_root=TRAIN_IMAGE_DIR,
            ground_truth_path=GROUND_TRUTH_TRAIN_PATH,
            audit_output_path=TRAIN_EXTRACTION_AUDIT_JSON_PATH,
        )
    if split in {"all", "prod"}:
        summaries["prod"] = extract_split_packages(
            split_name="prod",
            client_root=PROD_CLIENT_DIR,
            pdf_root=PROD_PDF_DIR,
            image_root=PROD_IMAGE_DIR,
            ground_truth_path=None,
            audit_output_path=PROD_EXTRACTION_AUDIT_JSON_PATH,
        )

    if DATA_SUMMARY_PATH.exists():
        existing_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing_summary = {}
    existing_summary["package_extraction_summary"] = summaries
    existing_summary["pipeline_mode"] = "direct_postgres"
    existing_summary["database_schema"] = "normalized_5_table_portfolio_schema"
    DATA_SUMMARY_PATH.write_text(json.dumps(existing_summary, indent=2), encoding="utf-8")

    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract structured claims from client intake packages.")
    parser.add_argument("--split", choices=["all", "train", "prod"], default="all")
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()
    main(split=args.split, reset_db=args.reset_db)
