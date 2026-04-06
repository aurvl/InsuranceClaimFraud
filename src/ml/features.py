from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DEFAULT_DECISION_THRESHOLD, SUSPICIOUS_PROVIDERS


REQUIRED_INPUT_COLUMNS = [
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

NUMERIC_MODEL_FEATURES = [
    "policy_age_days",
    "claim_amount",
    "customer_age",
    "num_previous_claims",
    "time_since_last_claim_days_filled",
    "missing_last_claim_flag",
    "claim_amount_to_policy_age",
    "recent_claim_flag",
    "suspicious_provider_flag",
    "high_amount_flag",
    "policy_recent_flag",
    "claim_month",
    "claim_quarter",
    "claim_weekday",
    "amount_per_previous_claim",
]

CATEGORICAL_MODEL_FEATURES = [
    "claim_type",
    "service_provider_id",
    "location",
    "weather_condition",
    "customer_age_band",
]


def prepare_claim_dataframe(dataframe: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    if isinstance(dataframe, dict):
        dataframe = pd.DataFrame([dataframe])
    elif isinstance(dataframe, list):
        dataframe = pd.DataFrame(dataframe)
    else:
        dataframe = dataframe.copy()

    for column in REQUIRED_INPUT_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = np.nan

    dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"], errors="coerce")
    dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"], errors="coerce")
    numeric_columns = [
        "policy_age_days",
        "claim_amount",
        "customer_age",
        "num_previous_claims",
        "time_since_last_claim_days",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe["claim_amount"] = dataframe["claim_amount"].astype(float)
    dataframe["claim_type"] = dataframe["claim_type"].astype("string")
    dataframe["service_provider_id"] = dataframe["service_provider_id"].astype("string")
    dataframe["location"] = dataframe["location"].astype("string")
    dataframe["weather_condition"] = dataframe["weather_condition"].astype("string")
    return dataframe


def assign_age_band(age: float | int | None) -> str:
    if pd.isna(age):
        return "unknown"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 50:
        return "35-49"
    if age < 65:
        return "50-64"
    return "65+"


def infer_risk_flags(
    dataframe: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
    suspicious_providers: list[str] | None = None,
    high_amount_threshold: float | None = None,
) -> list[list[str]]:
    df = prepare_claim_dataframe(dataframe)
    providers = set(suspicious_providers or SUSPICIOUS_PROVIDERS)
    if high_amount_threshold is None:
        high_amount_threshold = max(18_000.0, float(df["claim_amount"].fillna(0).quantile(0.9) if not df.empty else 18_000.0))

    flags_per_record: list[list[str]] = []
    for _, row in df.iterrows():
        record_flags: list[str] = []
        if float(row["claim_amount"]) >= high_amount_threshold:
            record_flags.append("high_claim_amount")
        if pd.notna(row["time_since_last_claim_days"]) and float(row["time_since_last_claim_days"]) <= 30:
            record_flags.append("rapid_repeat_claim")
        if pd.notna(row["policy_age_days"]) and float(row["policy_age_days"]) <= 45:
            record_flags.append("early_policy_claim")
        if str(row["service_provider_id"]) in providers:
            record_flags.append("suspicious_provider")
        if str(row["claim_type"]) == "hail_damage" and str(row["weather_condition"]) not in {"hail", "storm"}:
            record_flags.append("weather_claim_mismatch")
        if str(row["claim_type"]) == "water_damage" and str(row["weather_condition"]) == "clear":
            record_flags.append("weather_claim_mismatch")
        flags_per_record.append(record_flags)
    return flags_per_record


class ClaimFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        suspicious_providers: list[str] | None = None,
        high_amount_threshold: float | None = None,
    ) -> None:
        self.suspicious_providers = suspicious_providers
        self.high_amount_threshold = high_amount_threshold
    
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        dataframe = prepare_claim_dataframe(X)
        if self.high_amount_threshold is None:
            self.high_amount_threshold_ = float(max(18_000.0, dataframe["claim_amount"].quantile(0.9)))
        else:
            self.high_amount_threshold_ = float(self.high_amount_threshold)

        derived_suspicious_providers: set[str] = set(self.suspicious_providers or [])
        if y is not None:
            labels = pd.Series(y).reset_index(drop=True)
            provider_stats = (
                pd.concat(
                    [
                        dataframe.reset_index(drop=True)["service_provider_id"].rename("service_provider_id"),
                        labels.rename("is_fraud"),
                    ],
                    axis=1,
                )
                .dropna()
                .groupby("service_provider_id")
                .agg(total_claims=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"))
            )
            overall_rate = float(labels.mean())
            risky_providers = provider_stats.loc[
                (provider_stats["total_claims"] >= 30) & (provider_stats["fraud_rate"] >= overall_rate + 0.05)
            ].index.tolist()
            safe_providers = provider_stats.loc[
                provider_stats["total_claims"] >= 30
            ].sort_values(["fraud_rate", "total_claims"], ascending=[True, False]).index.tolist()
            derived_suspicious_providers.update(str(provider) for provider in risky_providers)
            self.safe_providers_ = [str(provider) for provider in safe_providers[:5]]
        else:
            self.safe_providers_ = []

        derived_suspicious_providers.update(SUSPICIOUS_PROVIDERS)
        self.suspicious_providers_ = sorted(derived_suspicious_providers)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dataframe = prepare_claim_dataframe(X)
        output = pd.DataFrame(index=dataframe.index)
        output["policy_age_days"] = dataframe["policy_age_days"]
        output["claim_amount"] = dataframe["claim_amount"]
        output["customer_age"] = dataframe["customer_age"]
        output["num_previous_claims"] = dataframe["num_previous_claims"].fillna(0)
        output["time_since_last_claim_days_filled"] = dataframe["time_since_last_claim_days"].fillna(9_999)
        output["missing_last_claim_flag"] = dataframe["time_since_last_claim_days"].isna().astype(int)
        output["claim_amount_to_policy_age"] = dataframe["claim_amount"] / dataframe["policy_age_days"].clip(lower=1)
        output["recent_claim_flag"] = (output["time_since_last_claim_days_filled"] <= 30).astype(int)
        output["suspicious_provider_flag"] = dataframe["service_provider_id"].isin(self.suspicious_providers_).astype(int)
        output["high_amount_flag"] = (dataframe["claim_amount"] >= self.high_amount_threshold_).astype(int)
        output["policy_recent_flag"] = (dataframe["policy_age_days"].fillna(9999) <= 45).astype(int)
        output["claim_month"] = dataframe["claim_date"].dt.month.fillna(0)
        output["claim_quarter"] = dataframe["claim_date"].dt.quarter.fillna(0)
        output["claim_weekday"] = dataframe["claim_date"].dt.weekday.fillna(0)
        output["amount_per_previous_claim"] = dataframe["claim_amount"] / (dataframe["num_previous_claims"].fillna(0) + 1)
        output["claim_type"] = dataframe["claim_type"].fillna("unknown")
        output["service_provider_id"] = dataframe["service_provider_id"].fillna("unknown")
        output["location"] = dataframe["location"].fillna("unknown")
        output["weather_condition"] = dataframe["weather_condition"].fillna("unknown")
        output["customer_age_band"] = dataframe["customer_age"].apply(assign_age_band)

        ordered_columns = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES
        return output.loc[:, ordered_columns]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ]
    )


def default_decision_threshold() -> float:
    return DEFAULT_DECISION_THRESHOLD
