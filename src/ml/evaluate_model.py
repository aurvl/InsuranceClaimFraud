from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_SUMMARY_PATH,
    EVALUATION_PATH,
    FEATURE_IMPORTANCE_PATH,
    HISTORICAL_TEST_PREDICTIONS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)
from src.ml.train_model import (
    DEFAULT_MIN_RECALL_FOR_THRESHOLD,
    RANDOM_STATE,
    TEST_SIZE,
    load_historical_training_data,
)


def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def load_model_bundle_from_path(model_path: Path) -> tuple[Any, dict[str, Any]]:
    metadata_path = MODEL_METADATA_PATH.with_name(f"{model_path.stem}{MODEL_METADATA_PATH.suffix}")
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def precision_recall_at_top_k(
    y_true: pd.Series,
    scores: pd.Series,
    top_fraction: float,
) -> dict[str, float]:
    threshold_index = max(1, int(len(scores) * top_fraction))
    top_records = (
        pd.DataFrame(
            {
                "y_true": pd.Series(y_true).reset_index(drop=True),
                "score": pd.Series(scores).reset_index(drop=True),
            }
        )
        .sort_values("score", ascending=False)
        .head(threshold_index)
    )
    true_positives = int(top_records["y_true"].sum())
    precision_at_k = true_positives / len(top_records) if len(top_records) else 0.0
    recall_at_k = true_positives / int(y_true.sum()) if int(y_true.sum()) else 0.0
    return {
        "fraction": top_fraction,
        "records": int(len(top_records)),
        "precision": round(float(precision_at_k), 6),
        "recall": round(float(recall_at_k), 6),
    }


def save_feature_importances(model) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return
    feature_names = preprocessor.get_feature_names_out()
    importance_df = (
        pd.DataFrame(
            {
                "feature_name": feature_names,
                "importance": classifier.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def compute_model_metrics(y_test: pd.Series, scores: pd.Series, decision_threshold: float) -> dict[str, Any]:
    predictions = (scores >= decision_threshold).astype(int)
    precision = round(float(precision_score(y_test, predictions, zero_division=0)), 6)
    recall = round(float(recall_score(y_test, predictions, zero_division=0)), 6)
    f1 = round(float(f1_score(y_test, predictions, zero_division=0)), 6)
    roc_auc = round(float(roc_auc_score(y_test, scores)), 6)
    deployment_score = round(precision * 0.65 + f1 * 0.25 + recall * 0.10, 6)
    return {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "decision_threshold": round(float(decision_threshold), 4),
        "deployment_score": deployment_score,
        "top_k": [
            precision_recall_at_top_k(y_true=y_test, scores=pd.Series(scores), top_fraction=0.01),
            precision_recall_at_top_k(y_true=y_test, scores=pd.Series(scores), top_fraction=0.05),
            precision_recall_at_top_k(y_true=y_test, scores=pd.Series(scores), top_fraction=0.10),
        ],
    }


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    recall_gate = 1.0 if float(metrics["recall"]) >= DEFAULT_MIN_RECALL_FOR_THRESHOLD else 0.0
    return (
        recall_gate,
        float(metrics["precision"]),
        float(metrics["f1"]),
        float(metrics["roc_auc"]),
    )


def evaluate_model(source: str = "postgres", ignore_missing_base_model: bool = False) -> dict[str, Any]:
    base_model_available = MODEL_PATH.exists() and MODEL_METADATA_PATH.exists()
    if not base_model_available and not ignore_missing_base_model:
        raise FileNotFoundError(
            "Base model or metadata not found. Train a champion model before running evaluation."
        )

    historical_df = load_historical_training_data(source=source)
    X = historical_df.drop(columns=["is_fraud"]).copy()
    y = historical_df["is_fraud"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results: dict[str, Any] = {
        "development_policy": {
            "production_data_used_in_training": False,
            "production_data_used_in_evaluation": False,
            "notes": (
                "Pseudo-production claims are reserved for post-development scoring only. "
                "Historical evaluation is performed on extracted client-package data loaded from PostgreSQL."
            ),
        },
        "model_selection_policy": {
            "objective": "maximize_precision_subject_to_minimum_recall",
            "minimum_recall": DEFAULT_MIN_RECALL_FOR_THRESHOLD,
        },
    }

    per_model_results: dict[str, Any] = {}
    best_model_name: str | None = None
    best_model_path: str | None = None
    best_model_metrics: dict[str, Any] | None = None
    best_model_instance = None

    model_paths = sorted(MODEL_PATH.parent.glob("candidate_*.pkl"))
    for model_path in model_paths:
        model_name = model_path.stem.replace("candidate_", "")
        model, metadata = load_model_bundle_from_path(model_path)
        scores = model.predict_proba(X_test)[:, 1]
        metrics = compute_model_metrics(y_test, scores, float(metadata.get("decision_threshold", 0.5)))
        metrics["selected_imbalance_strategy"] = metadata.get("selected_imbalance_strategy")
        metrics["validation_metrics"] = metadata.get("selected_validation_metrics")
        per_model_results[model_name] = metrics

        if best_model_metrics is None or selection_key(metrics) > selection_key(best_model_metrics):
            best_model_name = model_name
            best_model_path = str(model_path)
            best_model_metrics = metrics
            best_model_instance = model

    if base_model_available:
        base_model, base_metadata = load_model_bundle()
        base_scores = base_model.predict_proba(X_test)[:, 1]
        results["historical_holdout"] = compute_model_metrics(
            y_test,
            base_scores,
            float(base_metadata.get("decision_threshold", 0.5)),
        )
        results["deployment_champion_metadata"] = {
            "classifier": base_metadata.get("classifier"),
            "selected_imbalance_strategy": base_metadata.get("selected_imbalance_strategy"),
            "decision_threshold": base_metadata.get("decision_threshold"),
        }

    results["multi_model"] = per_model_results
    results["best_model"] = best_model_name
    results["best_model_path"] = best_model_path
    results["best_model_metrics"] = best_model_metrics

    if best_model_instance is not None and best_model_metrics is not None:
        best_scores = best_model_instance.predict_proba(X_test)[:, 1]
        best_threshold = float(best_model_metrics["decision_threshold"])
        best_predictions = (best_scores >= best_threshold).astype(int)

        historical_predictions_df = X_test.copy()
        historical_predictions_df["fraud_probability"] = best_scores
        historical_predictions_df["predicted_is_fraud"] = best_predictions
        historical_predictions_df["actual_is_fraud"] = y_test.to_numpy()
        historical_predictions_df.to_csv(HISTORICAL_TEST_PREDICTIONS_PATH, index=False)
        save_feature_importances(model=best_model_instance)

    EVALUATION_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if DATA_SUMMARY_PATH.exists():
        existing_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing_summary = {}
    existing_summary["evaluation_summary"] = results
    DATA_SUMMARY_PATH.write_text(json.dumps(existing_summary, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    logger.info("Evaluation completed for {} candidate models", len(per_model_results))
    return results


def main(source: str = "postgres", ignore_missing_base_model: bool = False) -> dict[str, Any]:
    return evaluate_model(source=source, ignore_missing_base_model=ignore_missing_base_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained fraud model.")
    parser.add_argument("--source", choices=["postgres"], default="postgres")
    parser.add_argument(
        "--ignore-missing-base-model",
        action="store_true",
        help="Skip base-model evaluation if fraud_model.pkl is missing.",
    )
    args = parser.parse_args()
    main(source=args.source, ignore_missing_base_model=args.ignore_missing_base_model)
