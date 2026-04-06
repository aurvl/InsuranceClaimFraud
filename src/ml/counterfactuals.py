from __future__ import annotations

import json
import pickle
from itertools import combinations
from typing import Any

import pandas as pd

from src.config import MODEL_METADATA_PATH, MODEL_PATH


def load_model_and_metadata() -> tuple[Any, dict[str, Any]]:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def score_claim(model: Any, claim_payload: dict[str, Any]) -> float:
    claim_df = pd.DataFrame([claim_payload])
    probability = float(model.predict_proba(claim_df)[0, 1])
    return probability


def update_policy_start_date(claim_payload: dict[str, Any], new_policy_age_days: int) -> dict[str, Any]:
    updated_payload = dict(claim_payload)
    claim_date = pd.to_datetime(updated_payload["claim_date"])
    updated_payload["policy_age_days"] = int(new_policy_age_days)
    updated_payload["policy_start_date"] = (
        claim_date - pd.Timedelta(days=int(new_policy_age_days))
    ).date().isoformat()
    return updated_payload


def candidate_safe_providers(metadata: dict[str, Any], original_provider: str) -> list[str]:
    providers = [str(provider) for provider in metadata.get("safe_providers", []) if str(provider) != original_provider]
    if providers:
        return providers
    suspicious = set(str(provider) for provider in metadata.get("suspicious_providers", []))
    fallback_providers = [f"SP-{index:03d}" for index in range(1, 121)]
    return [provider for provider in fallback_providers if provider not in suspicious and provider != original_provider][:3]


def build_candidate_claims(claim_payload: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    original_amount = float(claim_payload["claim_amount"])
    original_gap = claim_payload.get("time_since_last_claim_days")
    original_gap = None if pd.isna(original_gap) else int(original_gap)
    original_provider = str(claim_payload["service_provider_id"])
    original_policy_age = int(claim_payload["policy_age_days"])
    safe_providers = candidate_safe_providers(metadata=metadata, original_provider=original_provider)

    candidates: list[dict[str, Any]] = []

    amount_reductions = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
    for multiplier in amount_reductions:
        updated = dict(claim_payload)
        updated["claim_amount"] = round(original_amount * multiplier, 2)
        candidates.append(
            {
                "changes": {"claim_amount": updated["claim_amount"]},
                "payload": updated,
                "priority": 1,
            }
        )

    for target_gap in [30, 60, 90, 180]:
        if original_gap is None or target_gap > original_gap:
            updated = dict(claim_payload)
            updated["time_since_last_claim_days"] = target_gap
            candidates.append(
                {
                    "changes": {"time_since_last_claim_days": target_gap},
                    "payload": updated,
                    "priority": 1,
                }
            )

    for target_policy_age in [60, 90, 180, 365]:
        if target_policy_age > original_policy_age:
            updated = update_policy_start_date(claim_payload=claim_payload, new_policy_age_days=target_policy_age)
            candidates.append(
                {
                    "changes": {"policy_age_days": target_policy_age, "policy_start_date": updated["policy_start_date"]},
                    "payload": updated,
                    "priority": 1,
                }
            )

    for provider in safe_providers:
        updated = dict(claim_payload)
        updated["service_provider_id"] = provider
        candidates.append(
            {
                "changes": {"service_provider_id": provider},
                "payload": updated,
                "priority": 1,
            }
        )

    primitive_changes = candidates.copy()
    for first_candidate, second_candidate in combinations(primitive_changes[:12], 2):
        combined_payload = dict(claim_payload)
        combined_changes = dict(first_candidate["changes"])
        combined_changes.update(second_candidate["changes"])
        combined_payload.update(combined_changes)
        if "policy_age_days" in combined_changes:
            combined_payload = update_policy_start_date(
                claim_payload=combined_payload,
                new_policy_age_days=int(combined_changes["policy_age_days"]),
            )
        candidates.append(
            {
                "changes": combined_changes,
                "payload": combined_payload,
                "priority": 2,
            }
        )

    return candidates


def generate_counterfactual(
    claim_payload: dict[str, Any],
    model: Any | None = None,
    metadata: dict[str, Any] | None = None,
    decision_threshold: float | None = None,
) -> dict[str, Any]:
    model = model or load_model_and_metadata()[0]
    metadata = metadata or load_model_and_metadata()[1]
    threshold = float(decision_threshold or metadata.get("decision_threshold", 0.5))

    original_probability = score_claim(model=model, claim_payload=claim_payload)
    if original_probability < threshold:
        return {
            "status": "not_required",
            "heuristic": True,
            "original_probability": round(original_probability, 6),
            "decision_threshold": threshold,
            "message": "The claim is already below the fraud decision threshold.",
        }

    best_candidate: dict[str, Any] | None = None
    best_probability = original_probability

    for candidate in build_candidate_claims(claim_payload=claim_payload, metadata=metadata):
        candidate_probability = score_claim(model=model, claim_payload=candidate["payload"])
        if candidate_probability < threshold:
            if best_candidate is None:
                best_candidate = candidate
                best_probability = candidate_probability
                continue
            if candidate["priority"] < best_candidate["priority"]:
                best_candidate = candidate
                best_probability = candidate_probability
                continue
            if candidate["priority"] == best_candidate["priority"] and candidate_probability < best_probability:
                best_candidate = candidate
                best_probability = candidate_probability

    if best_candidate is None:
        return {
            "status": "not_found",
            "heuristic": True,
            "original_probability": round(original_probability, 6),
            "decision_threshold": threshold,
            "message": "No valid heuristic counterfactual was found with the current search space.",
        }

    original_values = {field: claim_payload.get(field) for field in best_candidate["changes"].keys()}
    return {
        "status": "found",
        "heuristic": True,
        "original_probability": round(original_probability, 6),
        "counterfactual_probability": round(best_probability, 6),
        "decision_threshold": threshold,
        "changes": {
            "original_values": original_values,
            "proposed_values": best_candidate["changes"],
        },
    }
