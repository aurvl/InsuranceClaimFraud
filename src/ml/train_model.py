from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.config import (
    DATA_SUMMARY_PATH,
    DEVELOPMENT_CUTOFF_DATE,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PostgresSettings,
    PRODUCTION_START_DATE,
)
from src.db.query_from_postgres import fetch_historical_claims
from src.ml.features import ClaimFeatureEngineer, build_preprocessor, default_decision_threshold


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
DEFAULT_MIN_RECALL_FOR_THRESHOLD = float(os.getenv("MODEL_MIN_RECALL_FOR_THRESHOLD", "0.20"))
def load_historical_training_data(source: str = "postgres") -> pd.DataFrame:
    if source != "postgres":
        raise ValueError("PostgreSQL is the only supported training source in this pipeline.")
    return fetch_historical_claims(settings=PostgresSettings.from_env())


def get_classifier(name: str, imbalance_strategy: str = "weighted", y_train: pd.Series | None = None):
    name = name.lower()
    if name in ("rf", "randomforest"):
        return RandomForestClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name in ("svm", "svc"):
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        )
    if name in ("xgb", "xgboost"):
        xgb_kwargs = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        }
        if imbalance_strategy == "weighted" and y_train is not None:
            negatives = int((pd.Series(y_train) == 0).sum())
            positives = int((pd.Series(y_train) == 1).sum())
            xgb_kwargs["scale_pos_weight"] = negatives / max(positives, 1)
        return XGBClassifier(**xgb_kwargs)
    raise ValueError(f"Unknown classifier: {name}")


def build_training_pipeline(
    classifier_name: str = "rf",
    imbalance_strategy: str = "weighted",
    y_train: pd.Series | None = None,
) -> Pipeline:
    clf = get_classifier(classifier_name, imbalance_strategy=imbalance_strategy, y_train=y_train)
    return Pipeline(
        steps=[
            ("feature_engineer", ClaimFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ]
    )


def parse_classifier_names(classifier_name: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(classifier_name, str) and classifier_name.lower() == "all":
        return ["rf", "svm", "xgb"]
    if isinstance(classifier_name, str) and "," in classifier_name:
        return [item.strip() for item in classifier_name.split(",") if item.strip()]
    if isinstance(classifier_name, (list, tuple)):
        return [str(item).strip() for item in classifier_name if str(item).strip()]
    return [str(classifier_name).strip()]


def candidate_model_path(classifier_name: str) -> Any:
    return MODEL_DIR / f"candidate_{classifier_name}.pkl"


def candidate_metadata_path(classifier_name: str) -> Any:
    return MODEL_DIR / f"candidate_{classifier_name}.json"


def candidate_strategies_for_classifier(classifier_name: str) -> list[str]:
    if classifier_name.lower() in {"svm", "svc"}:
        return ["weighted"]
    return ["weighted", "oversample"]


def oversample_minority_class(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    combined = X.copy().reset_index(drop=True)
    combined["_target"] = pd.Series(y).reset_index(drop=True)
    majority = combined.loc[combined["_target"] == 0]
    minority = combined.loc[combined["_target"] == 1]
    if majority.empty or minority.empty:
        return X.copy(), y.copy()
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=RANDOM_STATE,
    )
    balanced = (
        pd.concat([majority, minority_upsampled], axis=0)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    return balanced.drop(columns=["_target"]), balanced["_target"].astype(int)


def fit_pipeline_for_strategy(
    classifier_name: str,
    imbalance_strategy: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    if imbalance_strategy == "oversample":
        X_fit, y_fit = oversample_minority_class(X_train, y_train)
    else:
        X_fit, y_fit = X_train.copy(), y_train.copy()

    pipeline = build_training_pipeline(
        classifier_name=classifier_name,
        imbalance_strategy=imbalance_strategy,
        y_train=y_fit,
    )

    fit_kwargs: dict[str, Any] = {}
    if imbalance_strategy == "weighted" and classifier_name.lower() not in {"xgb", "xgboost"}:
        fit_kwargs["classifier__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_fit)

    pipeline.fit(X_fit, y_fit, **fit_kwargs)
    return pipeline, X_fit, y_fit


def compute_threshold_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    precision = float(precision_score(y_true, predictions, zero_division=0))
    recall = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "deployment_score": round(precision * 0.65 + f1 * 0.25 + recall * 0.10, 6),
    }


def select_decision_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    minimum_recall: float = DEFAULT_MIN_RECALL_FOR_THRESHOLD,
) -> dict[str, float]:
    quantile_thresholds = np.quantile(scores, [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98])
    threshold_grid = np.unique(
        np.round(
            np.concatenate([np.linspace(0.25, 0.95, 71), quantile_thresholds]),
            4,
        )
    )
    candidates = [compute_threshold_metrics(y_true, scores, float(threshold)) for threshold in threshold_grid]
    eligible = [candidate for candidate in candidates if candidate["recall"] >= minimum_recall]

    if eligible:
        return max(
            eligible,
            key=lambda item: (
                item["precision"],
                item["f1"],
                item["deployment_score"],
                item["threshold"],
            ),
        )

    return max(
        candidates,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["recall"],
            item["deployment_score"],
        ),
    )


def evaluate_candidate(
    pipeline: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    threshold: float,
) -> dict[str, float]:
    scores = pipeline.predict_proba(X_eval)[:, 1]
    metrics = compute_threshold_metrics(y_eval, scores, threshold)
    metrics["roc_auc"] = round(float(roc_auc_score(y_eval, scores)), 6)
    return metrics


def champion_ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = candidate["validation_metrics"]
    recall_gate = 1.0 if metrics["recall"] >= DEFAULT_MIN_RECALL_FOR_THRESHOLD else 0.0
    return (
        recall_gate,
        float(metrics["precision"]),
        float(metrics["f1"]),
        float(metrics["roc_auc"]),
    )


def save_model_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    classifier_name: str,
    *,
    champion: bool,
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    classifier_model_path = candidate_model_path(classifier_name)
    classifier_metadata_path = candidate_metadata_path(classifier_name)

    with classifier_model_path.open("wb") as model_file:
        pickle.dump(pipeline, model_file)
    classifier_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if champion:
        champion_metadata = dict(metadata)
        champion_metadata["artifact_role"] = "deployment_champion"
        with MODEL_PATH.open("wb") as model_file:
            pickle.dump(pipeline, model_file)
        MODEL_METADATA_PATH.write_text(json.dumps(champion_metadata, indent=2), encoding="utf-8")


def train_model(source: str = "postgres", classifier_name: str = "rf") -> dict[str, Any]:
    dataframe = load_historical_training_data(source=source)
    if dataframe.empty:
        raise ValueError("No historical training data available.")

    X = dataframe.drop(columns=["is_fraud"]).copy()
    y = dataframe["is_fraud"].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    classifier_names = parse_classifier_names(classifier_name)
    training_results: dict[str, dict[str, Any]] = {}
    champion_classifier_name: str | None = None
    champion_metadata: dict[str, Any] | None = None
    champion_pipeline: Pipeline | None = None

    for current_classifier_name in classifier_names:
        logger.info("Benchmarking classifier={} across imbalance strategies", current_classifier_name)
        strategy_results: list[dict[str, Any]] = []

        for imbalance_strategy in candidate_strategies_for_classifier(current_classifier_name):
            pipeline, X_fit, y_fit = fit_pipeline_for_strategy(
                classifier_name=current_classifier_name,
                imbalance_strategy=imbalance_strategy,
                X_train=X_train,
                y_train=y_train,
            )

            validation_scores = pipeline.predict_proba(X_val)[:, 1]
            tuned_threshold = select_decision_threshold(y_val, validation_scores)
            validation_metrics = evaluate_candidate(
                pipeline=pipeline,
                X_eval=X_val,
                y_eval=y_val,
                threshold=float(tuned_threshold["threshold"]),
            )
            training_accuracy = round(float(pipeline.score(X_fit, y_fit)), 6)

            strategy_results.append(
                {
                    "classifier": current_classifier_name,
                    "imbalance_strategy": imbalance_strategy,
                    "decision_threshold": float(tuned_threshold["threshold"]),
                    "training_rows_after_strategy": int(len(X_fit)),
                    "training_accuracy": training_accuracy,
                    "validation_metrics": validation_metrics,
                    "threshold_selection_policy": {
                        "objective": "maximize_precision_subject_to_minimum_recall",
                        "minimum_recall": DEFAULT_MIN_RECALL_FOR_THRESHOLD,
                    },
                }
            )

        best_strategy_result = max(strategy_results, key=champion_ranking_key)
        selected_strategy = best_strategy_result["imbalance_strategy"]
        selected_threshold = float(best_strategy_result["decision_threshold"])

        final_pipeline, X_final_fit, y_final_fit = fit_pipeline_for_strategy(
            classifier_name=current_classifier_name,
            imbalance_strategy=selected_strategy,
            X_train=X_dev,
            y_train=y_dev,
        )
        final_test_metrics = evaluate_candidate(
            pipeline=final_pipeline,
            X_eval=X_test,
            y_eval=y_test,
            threshold=selected_threshold,
        )

        feature_engineer: ClaimFeatureEngineer = final_pipeline.named_steps["feature_engineer"]
        metadata = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "validation_size_within_development": VALIDATION_SIZE,
            "decision_threshold": selected_threshold,
            "historical_rows": int(len(dataframe)),
            "development_rows": int(len(X_dev)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "final_training_rows_after_strategy": int(len(X_final_fit)),
            "historical_positive_rate": round(float(y.mean()), 6),
            "historical_date_min": pd.to_datetime(dataframe["claim_date"]).min().date().isoformat(),
            "historical_date_max": pd.to_datetime(dataframe["claim_date"]).max().date().isoformat(),
            "suspicious_providers": feature_engineer.suspicious_providers_,
            "safe_providers": feature_engineer.safe_providers_,
            "high_amount_threshold": round(float(feature_engineer.high_amount_threshold_), 2),
            "training_source": source,
            "historical_source_stage": "extracted_client_packages_loaded_to_postgres",
            "development_cutoff_date": DEVELOPMENT_CUTOFF_DATE.isoformat(),
            "production_start_date": PRODUCTION_START_DATE.isoformat(),
            "production_data_used_in_training": False,
            "classifier": current_classifier_name,
            "selected_imbalance_strategy": selected_strategy,
            "training_accuracy": round(float(final_pipeline.score(X_final_fit, y_final_fit)), 6),
            "validation_strategy_benchmark": strategy_results,
            "selected_validation_metrics": best_strategy_result["validation_metrics"],
            "holdout_test_metrics": final_test_metrics,
            "threshold_selection_policy": {
                "objective": "maximize_precision_subject_to_minimum_recall",
                "minimum_recall": DEFAULT_MIN_RECALL_FOR_THRESHOLD,
            },
            "model_path": str(candidate_model_path(current_classifier_name)),
        }

        training_results[current_classifier_name] = metadata
        logger.success(
            "Selected classifier={} strategy={} threshold={} val_precision={} holdout_precision={}",
            current_classifier_name,
            selected_strategy,
            selected_threshold,
            best_strategy_result["validation_metrics"]["precision"],
            final_test_metrics["precision"],
        )

        if champion_metadata is None or champion_ranking_key(
            {
                "validation_metrics": metadata["selected_validation_metrics"],
            }
        ) > champion_ranking_key({"validation_metrics": champion_metadata["selected_validation_metrics"]}):
            champion_classifier_name = current_classifier_name
            champion_metadata = metadata
            champion_pipeline = final_pipeline

    if champion_classifier_name is None or champion_metadata is None or champion_pipeline is None:
        raise RuntimeError("Training did not produce a champion model.")

    for current_classifier_name, metadata in training_results.items():
        champion = current_classifier_name == champion_classifier_name
        if champion:
            save_model_artifacts(champion_pipeline, metadata, current_classifier_name, champion=True)
        else:
            classifier_pipeline, _, _ = fit_pipeline_for_strategy(
                classifier_name=current_classifier_name,
                imbalance_strategy=metadata["selected_imbalance_strategy"],
                X_train=X_dev,
                y_train=y_dev,
            )
            save_model_artifacts(classifier_pipeline, metadata, current_classifier_name, champion=False)

    summary_payload = {
        "training_results": training_results,
        "deployment_champion": champion_classifier_name,
        "deployment_champion_model_path": str(MODEL_PATH),
        "deployment_champion_metadata_path": str(MODEL_METADATA_PATH),
    }

    if DATA_SUMMARY_PATH.exists():
        existing_summary = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing_summary = {}
    existing_summary.pop("training_summary", None)
    existing_summary["training_summaries"] = training_results
    existing_summary["deployment_champion"] = {
        "classifier": champion_classifier_name,
        "decision_threshold": champion_metadata["decision_threshold"],
        "selected_imbalance_strategy": champion_metadata["selected_imbalance_strategy"],
        "selected_validation_metrics": champion_metadata["selected_validation_metrics"],
        "holdout_test_metrics": champion_metadata["holdout_test_metrics"],
    }
    DATA_SUMMARY_PATH.write_text(json.dumps(existing_summary, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, indent=2))
    return summary_payload


def main(source: str = "postgres", classifier_name: str = "rf") -> dict[str, Any]:
    return train_model(source=source, classifier_name=classifier_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--source", choices=["postgres"], default="postgres")
    parser.add_argument(
        "--classifier",
        choices=["rf", "svm", "xgb", "all"],
        default="rf",
        help="Classifier to train (or 'all' to train all supported models)",
    )
    args = parser.parse_args()
    main(source=args.source, classifier_name=args.classifier)
