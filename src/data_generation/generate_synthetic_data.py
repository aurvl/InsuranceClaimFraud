from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    CLAIM_TYPES,
    DATA_SUMMARY_PATH,
    GROUND_TRUTH_PROD_PATH,
    GROUND_TRUTH_TRAIN_PATH,
    LOCATION_WEIGHTS,
    LOCATIONS,
    SUSPICIOUS_PROVIDERS,
    WEATHER_CONDITIONS,
    ensure_project_directories,
)


GLOBAL_POLICY_START = pd.Timestamp("2018-01-01")
GLOBAL_CLAIM_END = pd.Timestamp("2025-12-31")
SERVICE_PROVIDERS = [f"SP-{index:03d}" for index in range(1, 121)]

if len(LOCATION_WEIGHTS) != len(LOCATIONS):
    raise ValueError(
        "LOCATION_WEIGHTS must have the same length as LOCATIONS. "
        f"Got {len(LOCATION_WEIGHTS)} weights for {len(LOCATIONS)} locations."
    )


def cyclic_slice(values: list[str], start: int, length: int) -> list[str]:
    if not values:
        return []
    return [values[(start + offset) % len(values)] for offset in range(length)]


@dataclass(frozen=True)
class CustomerProfile:
    customer_id: str
    customer_age: int
    location: str
    policy_start_date: pd.Timestamp
    claim_offsets: list[int]
    base_risk: float


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_claim_counts(
    num_customers: int,
    target_claims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    claim_counts = rng.negative_binomial(2, 0.28, size=num_customers) + 1
    claim_counts = np.clip(claim_counts, 1, 16)
    while int(claim_counts.sum()) < target_claims:
        bump_indices = rng.choice(num_customers, size=max(10, num_customers // 20), replace=False)
        claim_counts[bump_indices] += 1
        claim_counts = np.clip(claim_counts, 1, 18)
    return claim_counts


def generate_claim_offsets(n_claims: int, rng: np.random.Generator) -> list[int]:
    offsets: list[int] = []
    current_offset = int(rng.integers(10, 150))
    offsets.append(current_offset)

    for index in range(1, n_claims):
        short_repeat_probability = min(0.12 + index * 0.02, 0.30)
        if rng.random() < short_repeat_probability:
            gap = int(rng.integers(7, 35))
        else:
            gap = int(rng.gamma(shape=2.8, scale=70.0)) + 25
        current_offset += gap
        offsets.append(current_offset)

    max_supported_offset = 2600
    if offsets[-1] > max_supported_offset:
        scale = max_supported_offset / offsets[-1]
        adjusted_offsets: list[int] = []
        last_offset = 5
        for raw_offset in offsets:
            scaled_offset = max(last_offset + 2, int(raw_offset * scale))
            adjusted_offsets.append(scaled_offset)
            last_offset = scaled_offset
        return adjusted_offsets
    return offsets


def generate_customer_profiles(
    num_customers: int,
    target_claims: int,
    rng: np.random.Generator,
) -> list[CustomerProfile]:
    claim_counts = generate_claim_counts(num_customers=num_customers, target_claims=target_claims, rng=rng)
    total_policy_window = (GLOBAL_CLAIM_END - GLOBAL_POLICY_START).days - 30
    customers: list[CustomerProfile] = []

    for index, claim_count in enumerate(claim_counts, start=1):
        claim_offsets = generate_claim_offsets(int(claim_count), rng)
        latest_policy_start = GLOBAL_CLAIM_END - pd.Timedelta(days=claim_offsets[-1] + 1)
        latest_policy_start = max(latest_policy_start, GLOBAL_POLICY_START)
        max_start_range = max(1, (latest_policy_start - GLOBAL_POLICY_START).days)
        start_day_offset = int(rng.integers(0, min(total_policy_window, max_start_range) + 1))
        policy_start_date = GLOBAL_POLICY_START + pd.Timedelta(days=start_day_offset)
        customers.append(
            CustomerProfile(
                customer_id=f"CUST-{index:05d}",
                customer_age=int(rng.integers(18, 86)),
                location=str(rng.choice(LOCATIONS, p=LOCATION_WEIGHTS)),
                policy_start_date=policy_start_date,
                claim_offsets=claim_offsets,
                base_risk=float(rng.normal(loc=0.0, scale=0.35)),
            )
        )

    return customers


def sample_weather(claim_date: pd.Timestamp, rng: np.random.Generator) -> str:
    month = claim_date.month
    if month in {12, 1, 2}:
        weights = np.array([0.18, 0.18, 0.10, 0.28, 0.08, 0.13, 0.05])
    elif month in {3, 4, 5}:
        weights = np.array([0.28, 0.22, 0.15, 0.06, 0.07, 0.08, 0.14])
    elif month in {6, 7, 8}:
        weights = np.array([0.36, 0.14, 0.17, 0.01, 0.04, 0.06, 0.22])
    else:
        weights = np.array([0.25, 0.24, 0.20, 0.04, 0.09, 0.12, 0.06])
    return str(rng.choice(WEATHER_CONDITIONS, p=weights))


def sample_claim_type(weather_condition: str, rng: np.random.Generator) -> str:
    if weather_condition in {"rain", "snow", "fog"}:
        weights = np.array([0.34, 0.10, 0.18, 0.14, 0.04, 0.12, 0.08])
    elif weather_condition in {"storm", "hail"}:
        weights = np.array([0.20, 0.05, 0.18, 0.18, 0.06, 0.05, 0.28])
    elif weather_condition == "heatwave":
        weights = np.array([0.23, 0.18, 0.14, 0.08, 0.09, 0.16, 0.12])
    else:
        weights = np.array([0.29, 0.18, 0.19, 0.09, 0.06, 0.13, 0.06])
    return str(rng.choice(CLAIM_TYPES, p=weights))


def base_claim_amount(claim_type: str) -> tuple[float, float]:
    mapping = {
        "collision": (8.25, 0.55),
        "theft": (9.00, 0.65),
        "windshield": (7.05, 0.40),
        "water_damage": (8.15, 0.60),
        "fire": (9.35, 0.65),
        "bodily_injury": (9.50, 0.55),
        "hail_damage": (7.55, 0.50),
    }
    return mapping[claim_type]


def sample_service_provider(
    location: str,
    claim_type: str,
    num_previous_claims: int,
    rng: np.random.Generator,
) -> str:
    base_location_index = LOCATIONS.index(location)
    preferred_start = (base_location_index * 9) % len(SERVICE_PROVIDERS)
    preferred_pool = cyclic_slice(SERVICE_PROVIDERS, preferred_start, 14)
    cross_location_pool = cyclic_slice(SERVICE_PROVIDERS, (preferred_start - 5) % len(SERVICE_PROVIDERS), 20)

    suspicious_bias = 0.18 if claim_type in {"collision", "theft"} or num_previous_claims >= 2 else 0.10
    if rng.random() < suspicious_bias:
        return str(rng.choice(SUSPICIOUS_PROVIDERS))

    if rng.random() < 0.75 and preferred_pool:
        return str(rng.choice(preferred_pool))
    return str(rng.choice(cross_location_pool))


def generate_claim_record(
    customer: CustomerProfile,
    claim_index: int,
    claim_date: pd.Timestamp,
    previous_claim_date: pd.Timestamp | None,
    rng: np.random.Generator,
) -> dict[str, Any]:
    num_previous_claims = claim_index
    time_since_last_claim_days = (
        None if previous_claim_date is None else int((claim_date - previous_claim_date).days)
    )
    policy_age_days = int((claim_date - customer.policy_start_date).days)
    weather_condition = sample_weather(claim_date=claim_date, rng=rng)
    claim_type = sample_claim_type(weather_condition=weather_condition, rng=rng)
    service_provider_id = sample_service_provider(
        location=customer.location,
        claim_type=claim_type,
        num_previous_claims=num_previous_claims,
        rng=rng,
    )

    amount_mu, amount_sigma = base_claim_amount(claim_type=claim_type)
    location_multiplier = 1.0 + (LOCATIONS.index(customer.location) / len(LOCATIONS)) * 0.08
    repeat_claim_multiplier = 1.0 + min(num_previous_claims, 4) * 0.05
    claim_amount = float(np.exp(rng.normal(amount_mu, amount_sigma)) * location_multiplier * repeat_claim_multiplier)
    claim_amount = min(max(claim_amount, 180.0), 85000.0)

    time_gap = 9999 if time_since_last_claim_days is None else time_since_last_claim_days
    suspicious_provider_flag = int(service_provider_id in SUSPICIOUS_PROVIDERS)
    unusual_combo_flag = int(
        (claim_type == "hail_damage" and weather_condition not in {"hail", "storm"})
        or (claim_type == "water_damage" and weather_condition == "clear")
        or (claim_type == "windshield" and claim_amount > 4500)
        or (claim_type == "theft" and claim_amount > 40000)
    )

    logit = -4.25
    logit += customer.base_risk
    logit += 0.000045 * claim_amount
    logit += 1.15 if claim_amount > 22000 else 0.0
    logit += 0.65 if claim_amount > 35000 else 0.0
    logit += 1.20 if policy_age_days < 30 else 0.0
    logit += 0.65 if policy_age_days < 90 else 0.0
    logit += 1.10 if time_gap <= 14 else 0.0
    logit += 0.55 if time_gap <= 45 else 0.0
    logit += 0.28 if num_previous_claims >= 3 else 0.0
    logit += 1.05 * suspicious_provider_flag
    logit += 0.70 * unusual_combo_flag
    logit += 0.35 if claim_type in {"theft", "bodily_injury"} and claim_amount > 18000 else 0.0
    logit += 0.22 if customer.customer_age < 23 or customer.customer_age > 77 else 0.0
    logit += 0.18 if customer.location in {"Paris", "Marseille", "Nice"} else 0.0
    logit += float(rng.normal(loc=0.0, scale=0.38))

    fraud_probability = float(sigmoid(logit))
    is_fraud = int(rng.binomial(1, fraud_probability))

    return {
        "claim_id": f"CLM-{customer.customer_id.split('-')[-1]}-{claim_index + 1:03d}",
        "customer_id": customer.customer_id,
        "claim_date": claim_date.date().isoformat(),
        "policy_start_date": customer.policy_start_date.date().isoformat(),
        "policy_age_days": policy_age_days,
        "claim_amount": round(claim_amount, 2),
        "claim_type": claim_type,
        "customer_age": customer.customer_age,
        "num_previous_claims": num_previous_claims,
        "time_since_last_claim_days": time_since_last_claim_days,
        "service_provider_id": service_provider_id,
        "location": customer.location,
        "weather_condition": weather_condition,
        "is_fraud": is_fraud,
        "fraud_propensity": round(fraud_probability, 6),
    }


def generate_claims_dataframe(
    num_customers: int = 9_000,
    target_claims: int = 60_000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = generate_customer_profiles(num_customers=num_customers, target_claims=target_claims, rng=rng)
    records: list[dict[str, Any]] = []

    for customer in customers:
        previous_claim_date: pd.Timestamp | None = None
        for claim_index, offset_days in enumerate(customer.claim_offsets):
            claim_date = customer.policy_start_date + pd.Timedelta(days=offset_days)
            claim_date = min(claim_date, GLOBAL_CLAIM_END)
            if previous_claim_date is not None and claim_date <= previous_claim_date:
                claim_date = previous_claim_date + pd.Timedelta(days=1)
                if claim_date > GLOBAL_CLAIM_END:
                    break
            records.append(
                generate_claim_record(
                    customer=customer,
                    claim_index=claim_index,
                    claim_date=claim_date,
                    previous_claim_date=previous_claim_date,
                    rng=rng,
                )
            )
            previous_claim_date = claim_date

    claims_df = pd.DataFrame.from_records(records)
    claims_df["claim_date"] = pd.to_datetime(claims_df["claim_date"])
    claims_df["policy_start_date"] = pd.to_datetime(claims_df["policy_start_date"])
    claims_df = claims_df.sort_values(["claim_date", "customer_id", "claim_id"]).reset_index(drop=True)

    if len(claims_df) < 50_000:
        raise ValueError("Synthetic data generation produced fewer than 50,000 claims.")

    return claims_df


def split_temporally(claims_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    counts_by_date = (
        claims_df.groupby(claims_df["claim_date"].dt.date, as_index=False)
        .size()
        .rename(columns={"claim_date": "date", "size": "count"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    counts_by_date["cumulative_ratio"] = counts_by_date["count"].cumsum() / len(claims_df)
    target_ratio = 0.90
    valid_cutoffs = counts_by_date.iloc[:-1].copy()
    closest_index = (valid_cutoffs["cumulative_ratio"] - target_ratio).abs().idxmin()
    cutoff_row = valid_cutoffs.loc[closest_index]
    cutoff_date = pd.Timestamp(cutoff_row["date"])

    train_df = claims_df.loc[claims_df["claim_date"] <= cutoff_date].copy()
    prod_df = claims_df.loc[claims_df["claim_date"] > cutoff_date].copy()

    if train_df.empty or prod_df.empty:
        raise ValueError("Temporal split failed to produce both historical and recent subsets.")
    if train_df["claim_date"].max() >= prod_df["claim_date"].min():
        raise ValueError("Production data is not strictly more recent than training data.")

    return train_df, prod_df, cutoff_date.date().isoformat()


def persist_outputs(train_df: pd.DataFrame, prod_df: pd.DataFrame, cutoff_date: str) -> dict[str, Any]:
    ensure_project_directories()

    export_columns = [
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

    train_export = train_df.loc[:, export_columns].copy()
    prod_export = prod_df.loc[:, export_columns].copy()

    train_export["claim_date"] = train_export["claim_date"].dt.date.astype(str)
    train_export["policy_start_date"] = train_export["policy_start_date"].dt.date.astype(str)
    prod_export["claim_date"] = prod_export["claim_date"].dt.date.astype(str)
    prod_export["policy_start_date"] = prod_export["policy_start_date"].dt.date.astype(str)

    train_export.to_json(GROUND_TRUTH_TRAIN_PATH, orient="records", indent=2, force_ascii=False)
    prod_export.to_json(GROUND_TRUTH_PROD_PATH, orient="records", indent=2, force_ascii=False)

    summary = {
        "total_claims": int(len(train_export) + len(prod_export)),
        "historical_claims": int(len(train_export)),
        "recent_claims": int(len(prod_export)),
        "historical_ratio": round(len(train_export) / (len(train_export) + len(prod_export)), 4),
        "recent_ratio": round(len(prod_export) / (len(train_export) + len(prod_export)), 4),
        "split_cutoff_date": cutoff_date,
        "historical_max_date": train_export["claim_date"].max(),
        "recent_min_date": prod_export["claim_date"].min(),
        "historical_fraud_rate": round(float(train_export["is_fraud"].mean()), 4),
        "recent_fraud_rate": round(float(prod_export["is_fraud"].mean()), 4),
        "historical_customers": int(train_export["customer_id"].nunique()),
        "recent_customers": int(prod_export["customer_id"].nunique()),
        "suspicious_providers": SUSPICIOUS_PROVIDERS,
        "claim_types": CLAIM_TYPES,
        "locations": LOCATIONS,
        "weather_conditions": WEATHER_CONDITIONS,
        "ground_truth_outputs": {
            "historical": str(GROUND_TRUTH_TRAIN_PATH),
            "production": str(GROUND_TRUTH_PROD_PATH),
        },
        "structured_data_origin": (
            "Canonical train/prod analytical rows are generated downstream by extracting and consolidating "
            "client packages (json + pdf + image) and inserting the structured claims directly into PostgreSQL."
        ),
    }

    DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(num_customers: int = 9_000, target_claims: int = 60_000, seed: int = 42) -> dict[str, Any]:
    claims_df = generate_claims_dataframe(
        num_customers=num_customers,
        target_claims=target_claims,
        seed=seed,
    )
    train_df, prod_df, cutoff_date = split_temporally(claims_df=claims_df)
    summary = persist_outputs(train_df=train_df, prod_df=prod_df, cutoff_date=cutoff_date)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic insurance claim data.")
    parser.add_argument("--num-customers", type=int, default=9_000)
    parser.add_argument("--target-claims", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(num_customers=args.num_customers, target_claims=args.target_claims, seed=args.seed)
