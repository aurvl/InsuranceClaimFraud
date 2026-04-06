from __future__ import annotations

import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import PRODUCTION_PACKAGE_DIR, PRODUCTION_PACKAGE_RECORD_DIR, ensure_project_directories


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def attachment_metadata(
    *,
    filename: str,
    content_type: str | None,
    content_bytes: bytes,
    attachment_kind: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kind": attachment_kind,
        "filename": filename,
        "content_type": content_type or "application/octet-stream",
        "size_bytes": len(content_bytes),
        "sha256": hashlib.sha256(content_bytes).hexdigest(),
    }
    if attachment_kind == "image":
        with Image.open(BytesIO(content_bytes)) as image:
            metadata["image_width"] = int(image.width)
            metadata["image_height"] = int(image.height)
            metadata["image_format"] = image.format
    return metadata


def package_directory_for_claim(claim_id: str) -> Path:
    ensure_project_directories()
    package_directory = PRODUCTION_PACKAGE_DIR / safe_name(claim_id)
    package_directory.mkdir(parents=True, exist_ok=True)
    return package_directory


def persist_production_package(
    *,
    claim_payload: dict[str, Any],
    claim_json_bytes: bytes,
    pdf_upload: tuple[str, str | None, bytes] | None = None,
    image_upload: tuple[str, str | None, bytes] | None = None,
    document_extraction_mode: str = "implemented_rule_based_pdf_and_image_extraction",
) -> dict[str, Any]:
    claim_id = str(claim_payload["claim_id"])
    package_directory = package_directory_for_claim(claim_id)

    claim_json_path = package_directory / "claim_payload.json"
    claim_json_path.write_bytes(claim_json_bytes)

    attachments: list[dict[str, Any]] = []
    stored_files: dict[str, str] = {"claim_json": str(claim_json_path)}

    for attachment_kind, upload in (("pdf", pdf_upload), ("image", image_upload)):
        if not upload:
            continue
        filename, content_type, content_bytes = upload
        stored_filename = f"{attachment_kind}_{safe_name(filename)}"
        stored_path = package_directory / stored_filename
        stored_path.write_bytes(content_bytes)
        metadata = attachment_metadata(
            filename=filename,
            content_type=content_type,
            content_bytes=content_bytes,
            attachment_kind=attachment_kind,
        )
        metadata["stored_path"] = str(stored_path)
        attachments.append(metadata)
        stored_files[attachment_kind] = str(stored_path)

    return {
        "claim_id": claim_id,
        "package_directory": str(package_directory),
        "stored_files": stored_files,
        "attachments": attachments,
        "document_extraction_mode": document_extraction_mode,
    }


def production_record_path(claim_id: str) -> Path:
    ensure_project_directories()
    PRODUCTION_PACKAGE_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    return PRODUCTION_PACKAGE_RECORD_DIR / f"{safe_name(claim_id)}.json"


def persist_production_record(record: dict[str, Any]) -> str:
    record_path = production_record_path(str(record["claim_payload"]["claim_id"]))
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(record_path)


def load_production_record(claim_id: str) -> dict[str, Any] | None:
    record_path = production_record_path(claim_id)
    if not record_path.exists():
        return None
    return json.loads(record_path.read_text(encoding="utf-8"))
