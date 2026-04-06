from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path #noqa
from typing import Any

import pandas as pd

from src.config import DATA_SUMMARY_PATH, MODEL_METADATA_PATH, MODEL_PATH
from src.db.load_to_postgres import upsert_production_claim_decisions
from src.db.query_from_postgres import fetch_production_intake_claims
def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def score_production_claims(include_offline_labels: bool = False) -> dict[str, Any]:
    """Score production claims using the trained model without relying on any production labels, and store decisions in PostgreSQL."""
    model, metadata = load_model_bundle()
    production_df = fetch_production_intake_claims()
    if production_df.empty:
        raise ValueError("No production intake claims found in PostgreSQL.")
    scoring_input = production_df.drop(
        columns=["chosen_sources", "mismatches", "package_metadata", "structured_claim", "created_at", "updated_at"],
        errors="ignore",
    )
    probabilities = model.predict_proba(scoring_input)[:, 1]
    threshold = float(metadata.get("decision_threshold", 0.5))
    predictions = (probabilities >= threshold).astype(int)

    output_df = scoring_input.copy()
    output_df["fraud_probability"] = probabilities
    output_df["predicted_is_fraud"] = predictions
    if include_offline_labels:
        try:
            from src.config import GROUND_TRUTH_PROD_PATH

            offline_labels = pd.read_json(GROUND_TRUTH_PROD_PATH, orient="records")[["claim_id", "is_fraud"]]
            output_df = output_df.merge(offline_labels, on="claim_id", how="left").rename(
                columns={"is_fraud": "offline_actual_is_fraud"}
            )
        except Exception:
            pass

    decisions_df = output_df.copy()
    decisions_df["model_decision"] = decisions_df["fraud_probability"].apply(
        lambda score: "fraud" if float(score) >= threshold else "non_fraud"
    )
    decisions_df["decision_threshold"] = threshold
    decisions_df["intake_mode"] = "batch_postgres_scoring"
    decisions_df["package_directory"] = None
    decisions_df["record_path"] = None
    decisions_df["chosen_sources"] = [{} for _ in range(len(decisions_df))]
    decisions_df["mismatch_fields"] = [[] for _ in range(len(decisions_df))]
    decisions_df["package_metadata"] = [{} for _ in range(len(decisions_df))]
    decisions_df["prediction_metadata"] = [
        {
            "scoring_input_stage": "production_intake_postgres",
            "offline_labels_included": include_offline_labels,
        }
        for _ in range(len(decisions_df))
    ]
    upsert_production_claim_decisions(decisions_df)

    summary = {
        "scored_rows": int(len(output_df)),
        "decision_threshold": threshold,
        "production_labels_used_for_scoring": False,
        "offline_labels_included_in_output": include_offline_labels,
        "scoring_input_stage": "extracted_production_packages",
        "stored_in_postgres_table": metadata.get("production_table_name", "production_claim_decisions"),
    }
    if DATA_SUMMARY_PATH.exists():
        existing_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing_summary = {}
    existing_summary["production_scoring_summary"] = summary
    DATA_SUMMARY_PATH.write_text(json.dumps(existing_summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def main(include_offline_labels: bool = False) -> dict[str, Any]:
    return score_production_claims(include_offline_labels=include_offline_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score production claims without using production labels in development."
    )
    parser.add_argument("--include-offline-labels", action="store_true")
    args = parser.parse_args()
    main(include_offline_labels=args.include_offline_labels)
