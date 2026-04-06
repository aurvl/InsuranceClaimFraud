from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from src.config import (
    DATA_SUMMARY_PATH,
    GROUND_TRUTH_PROD_PATH,
    GROUND_TRUTH_TRAIN_PATH,
    PROD_CLIENT_DIR,
    PROD_IMAGE_DIR,
    PROD_PDF_DIR,
    TRAIN_CLIENT_DIR,
    TRAIN_IMAGE_DIR,
    TRAIN_PDF_DIR,
    ensure_project_directories,
)


def load_claims(json_path: Path) -> pd.DataFrame:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    dataframe = pd.DataFrame.from_records(payload)
    if dataframe.empty:
        raise ValueError(f"No claims found in {json_path}.")
    dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"])
    dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"])
    return dataframe


def ensure_customer_directories(dataframe: pd.DataFrame, root_directory: Path) -> None:
    for customer_id in dataframe["customer_id"].dropna().unique():
        (root_directory / str(customer_id)).mkdir(parents=True, exist_ok=True)


def cleanup_artifact_directory(directory: Path, pattern: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for artifact_path in directory.glob(pattern):
        artifact_path.unlink(missing_ok=True)


def cleanup_customer_artifacts(root_directory: Path) -> None:
    for artifact_path in root_directory.rglob("claim_*.*"):
        artifact_path.unlink(missing_ok=True)


def select_document_subset(dataframe: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or len(dataframe) <= sample_size:
        return dataframe.sort_values("claim_date").reset_index(drop=True)
    return dataframe.sample(n=sample_size, random_state=seed).sort_values("claim_date").reset_index(drop=True)


def build_claim_summary_lines(row: pd.Series) -> list[str]:
    lines = [
        "Insurance Claim Summary",
        "",
        f"Claim ID: {row['claim_id']}",
        f"Customer ID: {row['customer_id']}",
        f"Claim Date: {row['claim_date'].date().isoformat()}",
        f"Policy Start Date: {row['policy_start_date'].date().isoformat()}",
        f"Policy Age (days): {int(row['policy_age_days'])}",
        f"Claim Type: {row['claim_type']}",
        f"Claim Amount: EUR {float(row['claim_amount']):,.2f}",
        f"Customer Age: {int(row['customer_age'])}",
        f"Previous Claims: {int(row['num_previous_claims'])}",
        (
            "Time Since Last Claim: N/A"
            if pd.isna(row["time_since_last_claim_days"])
            else f"Time Since Last Claim: {int(row['time_since_last_claim_days'])} days"
        ),
        f"Service Provider: {row['service_provider_id']}",
        f"Location: {row['location']}",
        f"Weather: {row['weather_condition']}",
        "",
        "Synthetic adjuster note:",
        (
            f"The customer reported a {row['claim_type']} incident in {row['location']} under "
            f"{row['weather_condition']} conditions. The requested amount is EUR {float(row['claim_amount']):,.2f}. "
            "This package is generated locally to emulate a claim intake workflow with CRM data and attachments."
        ),
    ]
    return lines


def create_pdf_artifact(row: pd.Series, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_canvas = canvas.Canvas(str(pdf_path), pagesize=letter)
    y_position = 760

    for line in build_claim_summary_lines(row):
        if y_position < 70:
            pdf_canvas.showPage()
            y_position = 760
        pdf_canvas.drawString(48, y_position, line)
        y_position -= 22

    pdf_canvas.save()


def create_placeholder_damage_image(row: pd.Series, image_path: Path, rng: np.random.Generator) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    palette = {
        "collision": ((235, 240, 245), (201, 64, 64)),
        "theft": ((228, 233, 242), (48, 67, 108)),
        "windshield": ((224, 238, 242), (92, 159, 204)),
        "water_damage": ((222, 235, 248), (34, 119, 179)),
        "fire": ((252, 232, 224), (225, 105, 41)),
        "bodily_injury": ((248, 233, 233), (160, 54, 54)),
        "hail_damage": ((235, 240, 251), (79, 105, 173)),
    }
    background_color, accent_color = palette.get(row["claim_type"], ((240, 240, 240), (90, 90, 90)))
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((42, 88, 598, 295), radius=28, fill=(250, 250, 252), outline=accent_color, width=4)
    draw.rectangle((92, 170, 220, 225), fill=(55, 60, 72), outline=accent_color, width=3)
    draw.polygon([(210, 170), (275, 130), (430, 130), (490, 170)], fill=(78, 84, 96), outline=accent_color)
    draw.rectangle((275, 170, 490, 225), fill=(72, 79, 92), outline=accent_color, width=3)
    draw.ellipse((135, 218, 205, 290), fill=(35, 35, 35), outline=(20, 20, 20), width=3)
    draw.ellipse((395, 218, 465, 290), fill=(35, 35, 35), outline=(20, 20, 20), width=3)

    damage_count = 6 if row["claim_type"] in {"collision", "fire", "bodily_injury"} else 4
    for _ in range(damage_count):
        x1 = int(rng.integers(160, 450))
        y1 = int(rng.integers(135, 235))
        x2 = x1 + int(rng.integers(14, 55))
        y2 = y1 + int(rng.integers(8, 32))
        draw.line((x1, y1, x2, y2), fill=accent_color, width=int(rng.integers(2, 5)))
        if rng.random() < 0.4:
            draw.arc((x1 - 10, y1 - 10, x2 + 10, y2 + 10), start=15, end=210, fill=accent_color, width=3)

    font = ImageFont.load_default()
    draw.text((55, 32), f"{row['claim_id']} | {row['customer_id']}", fill=(30, 30, 30), font=font)
    draw.text(
        (55, 322),
        f"{row['claim_type']} | {row['location']} | {row['weather_condition']} | EUR {float(row['claim_amount']):,.0f}",
        fill=(35, 35, 35),
        font=font,
    )
    draw.text((430, 32), "Synthetic Damage View", fill=accent_color, font=font)

    image.save(image_path, format="PNG", optimize=True)


def create_claim_package_json(row: pd.Series, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "claim_id": row["claim_id"],
        "customer_id": row["customer_id"],
        "claim_date": row["claim_date"].date().isoformat(),
        "policy_start_date": row["policy_start_date"].date().isoformat(),
        "policy_age_days": int(row["policy_age_days"]),
        "claim_amount": float(row["claim_amount"]),
        "claim_type": row["claim_type"],
        "customer_age": int(row["customer_age"]),
        "num_previous_claims": int(row["num_previous_claims"]),
        "time_since_last_claim_days": (
            None if pd.isna(row["time_since_last_claim_days"]) else int(row["time_since_last_claim_days"])
        ),
        "service_provider_id": row["service_provider_id"],
        "location": row["location"],
        "weather_condition": row["weather_condition"],
        "crm_context": {
            "source_system": "synthetic_crm",
            "package_role": "structured_intake_payload",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def create_artifacts_for_split(
    dataframe: pd.DataFrame,
    customer_root: Path,
    pdf_root: Path,
    image_root: Path,
    document_sample_size: int,
    seed: int,
) -> dict[str, int | float]:
    ensure_customer_directories(dataframe=dataframe, root_directory=customer_root)
    cleanup_artifact_directory(pdf_root, "*.pdf")
    cleanup_artifact_directory(image_root, "*.png")
    cleanup_customer_artifacts(customer_root)

    document_claim_ids = set(
        select_document_subset(dataframe=dataframe, sample_size=document_sample_size, seed=seed)["claim_id"].tolist()
    )
    rng = np.random.default_rng(seed)

    for _, row in dataframe.sort_values("claim_date").iterrows():
        filename_stem = f"claim_{row['claim_id']}_{row['customer_id']}"
        customer_directory = customer_root / str(row["customer_id"])
        json_path = customer_directory / f"{filename_stem}.json"
        create_claim_package_json(row=row, json_path=json_path)

        if row["claim_id"] not in document_claim_ids:
            continue

        pdf_path = pdf_root / f"{filename_stem}.pdf"
        image_path = image_root / f"{filename_stem}.png"
        create_pdf_artifact(row=row, pdf_path=pdf_path)
        create_placeholder_damage_image(row=row, image_path=image_path, rng=rng)
        link_or_copy(pdf_path, customer_directory / pdf_path.name)
        link_or_copy(image_path, customer_directory / image_path.name)

    pdf_count = int(len(list(pdf_root.glob("*.pdf"))))
    image_count = int(len(list(image_root.glob("*.png"))))
    json_count = int(len(list(customer_root.rglob("claim_*.json"))))
    total_claims = int(len(dataframe))
    return {
        "customers": int(dataframe["customer_id"].nunique()),
        "claims": total_claims,
        "json_packages": json_count,
        "pdf_artifacts": pdf_count,
        "image_artifacts": image_count,
        "document_coverage_ratio": round(pdf_count / total_claims, 4) if total_claims else 0.0,
        "full_package_count": min(pdf_count, image_count, json_count),
    }


def main(
    train_document_sample_size: int = 20,
    prod_document_sample_size: int = 10,
    seed: int = 42,
) -> dict[str, dict[str, int | float]]:
    ensure_project_directories()
    train_df = load_claims(GROUND_TRUTH_TRAIN_PATH)
    prod_df = load_claims(GROUND_TRUTH_PROD_PATH)

    summary = {
        "train": create_artifacts_for_split(
            dataframe=train_df,
            customer_root=TRAIN_CLIENT_DIR,
            pdf_root=TRAIN_PDF_DIR,
            image_root=TRAIN_IMAGE_DIR,
            document_sample_size=train_document_sample_size,
            seed=seed,
        ),
        "prod": create_artifacts_for_split(
            dataframe=prod_df,
            customer_root=PROD_CLIENT_DIR,
            pdf_root=PROD_PDF_DIR,
            image_root=PROD_IMAGE_DIR,
            document_sample_size=prod_document_sample_size,
            seed=seed + 7,
        ),
    }

    if DATA_SUMMARY_PATH.exists():
        existing_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing_summary = {}
    existing_summary["artifact_summary"] = summary
    existing_summary["raw_package_design"] = {
        "historical_json_packages_cover_all_claims": True,
        "historical_pdf_image_document_generation_is_sampled": train_document_sample_size > 0,
        "production_packages_include_documents_for_all_claims": prod_document_sample_size <= 0,
        "note": (
            "Structured train/prod claims are loaded directly into PostgreSQL by the extraction stage from the client package directories."
        ),
    }
    DATA_SUMMARY_PATH.write_text(json.dumps(existing_summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate local client packages with JSON, PDFs, and images.")
    parser.add_argument("--train-document-sample-size", type=int, default=20)
    parser.add_argument("--prod-document-sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        train_document_sample_size=args.train_document_sample_size,
        prod_document_sample_size=args.prod_document_sample_size,
        seed=args.seed,
    )
