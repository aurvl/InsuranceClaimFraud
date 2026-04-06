from __future__ import annotations

import json
import pickle
from contextlib import asynccontextmanager
from datetime import date
from io import StringIO
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.claim_package import load_production_record, persist_production_package, persist_production_record
from src.api.document_extraction import extract_and_consolidate_claim_package
from src.config import (
    DEVELOPMENT_CUTOFF_DATE,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PRODUCTION_START_DATE,
    PostgresSettings,
)
from src.db.load_to_postgres import (
    create_supporting_tables,
    upsert_production_claim_decision,
    upsert_production_claim_intake,
)
from src.db.postgres_connection import test_postgres_connection
from src.db.query_from_postgres import (
    fetch_claim_by_id,
    fetch_historical_claims,
    fetch_production_intake_by_id,
    fetch_production_decision_by_id,
)
from src.ml.counterfactuals import generate_counterfactual
from src.ml.features import infer_risk_flags


class ClaimPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim_id: str | None = Field(default=None)
    customer_id: str | None = Field(default=None)
    claim_date: date
    policy_start_date: date
    policy_age_days: int = Field(ge=0)
    claim_amount: float = Field(gt=0)
    claim_type: str
    customer_age: int = Field(ge=18, le=100)
    num_previous_claims: int = Field(ge=0)
    time_since_last_claim_days: int | None = Field(default=None, ge=0)
    service_provider_id: str
    location: str
    weather_condition: str

    @model_validator(mode="after")
    def validate_temporal_consistency(self) -> "ClaimPayload":
        computed_policy_age = (self.claim_date - self.policy_start_date).days
        if computed_policy_age != self.policy_age_days:
            raise ValueError("policy_age_days must equal claim_date - policy_start_date in days.")
        return self


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    claim: ClaimPayload | None = None
    claim_id: str | None = None
    decision_threshold: float | None = Field(default=None, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_input(self) -> "CounterfactualRequest":
        if not self.claim and not self.claim_id:
            raise ValueError("Provide either a claim payload or a claim_id.")
        return self


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On application startup, load the trained model and metadata, establish a connection to PostgreSQL, and create supporting tables if they don't exist. Store these resources in the application state for use in request handlers. On shutdown, clean up any resources if necessary.
    """
    
    postgres_settings = PostgresSettings.from_env()
    if not test_postgres_connection(settings=postgres_settings):
        raise RuntimeError("FastAPI startup failed because PostgreSQL is unreachable.")
    create_supporting_tables(settings=postgres_settings)

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))

    app.state.postgres_settings = postgres_settings
    app.state.model = model
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Insurance Claim Counterfactual Simulator API",
    version="2.0.0",
    lifespan=lifespan,
)


def serialize_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []
    serializable_df = dataframe.copy()
    for column in serializable_df.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable_df[column]):
            serializable_df[column] = serializable_df[column].dt.date.astype(str)
    return serializable_df.where(pd.notnull(serializable_df), None).to_dict(orient="records")


def ensure_claim_id(claim_payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(claim_payload)
    if not output.get("claim_id"):
        output["claim_id"] = f"INTAKE-{uuid4().hex[:10].upper()}"
    return output


def validate_production_claim_date(claim_payload: dict[str, Any]) -> None:
    claim_date_value = pd.to_datetime(claim_payload["claim_date"]).date()
    if claim_date_value < PRODUCTION_START_DATE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Production intake only accepts claims dated on or after {PRODUCTION_START_DATE.isoformat()}. "
                f"Received claim_date={claim_date_value.isoformat()}, which falls before the post-development boundary."
            ),
        )


def score_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims_df = pd.DataFrame(claims)
    probabilities = app.state.model.predict_proba(claims_df)[:, 1]
    threshold = float(app.state.metadata.get("decision_threshold", 0.5))
    risk_flags = infer_risk_flags(
        claims_df,
        suspicious_providers=app.state.metadata.get("suspicious_providers", []),
        high_amount_threshold=app.state.metadata.get("high_amount_threshold"),
    )

    results: list[dict[str, Any]] = []
    for claim, probability, flags in zip(claims, probabilities, risk_flags):
        results.append(
            {
                "claim_id": claim.get("claim_id"),
                "fraud_probability": round(float(probability), 6),
                "model_decision": "fraud" if float(probability) >= threshold else "non_fraud",
                "decision_threshold": threshold,
                "risk_flags": flags,
            }
        )
    return results


def parse_predict_payload(payload: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(payload, list):
        claims = [ensure_claim_id(ClaimPayload.model_validate(item).model_dump(mode="json")) for item in payload]
        return claims, True
    claim = ensure_claim_id(ClaimPayload.model_validate(payload).model_dump(mode="json"))
    return [claim], False


def query_filters(
    start_date: date | None,
    end_date: date | None,
    location: str | None,
    claim_type: str | None,
    is_fraud: bool | None,
    provider: str | None,
) -> dict[str, Any]:
    return {
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "location": location,
        "claim_type": claim_type,
        "is_fraud": is_fraud,
        "provider": provider,
    }


def production_policy_metadata() -> dict[str, Any]:
    return {
        "development_cutoff_date": DEVELOPMENT_CUTOFF_DATE.isoformat(),
        "production_start_date": PRODUCTION_START_DATE.isoformat(),
        "production_data_used_in_training": False,
        "production_data_used_in_evaluation": False,
        "document_extraction_mode": "implemented_rule_based_pdf_and_image_extraction",
        "historical_analytics_source": "postgres_from_extracted_client_packages",
    }


async def extract_claim_json_bytes(
    claim_json: str | None,
    claim_json_file: UploadFile | None,
) -> bytes:
    if claim_json_file is not None:
        return await claim_json_file.read()
    if claim_json is not None:
        return claim_json.encode("utf-8")
    raise HTTPException(status_code=400, detail="Provide claim_json text or a claim_json_file.")


async def ingest_and_score_package(
    claim_json: str | None,
    claim_json_file: UploadFile | None,
    pdf_file: UploadFile | None,
    image_file: UploadFile | None,
) -> dict[str, Any]:
    if pdf_file is None and image_file is None:
        raise HTTPException(
            status_code=400,
            detail="At least one attachment is required for package intake. Use /predict for structured-only requests.",
        )

    claim_json_bytes = await extract_claim_json_bytes(claim_json=claim_json, claim_json_file=claim_json_file)
    pdf_bytes = None if pdf_file is None else await pdf_file.read()
    image_bytes = None if image_file is None else await image_file.read()
    extraction_result = extract_and_consolidate_claim_package(
        claim_json_bytes=claim_json_bytes,
        pdf_bytes=pdf_bytes,
        image_filename=None if image_file is None else (image_file.filename or "damage.png"),
        image_bytes=image_bytes,
    )
    if extraction_result["missing_fields"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Document extraction could not build a complete structured claim. "
                f"Missing fields: {', '.join(extraction_result['missing_fields'])}"
            ),
        )

    claim_payload = ensure_claim_id(
        ClaimPayload.model_validate(extraction_result["structured_claim"]).model_dump(mode="json")
    )
    extraction_result["structured_claim"] = claim_payload
    validate_production_claim_date(claim_payload=claim_payload)

    pdf_upload = None
    if pdf_file is not None and pdf_bytes is not None:
        pdf_upload = (pdf_file.filename or "claim.pdf", pdf_file.content_type, pdf_bytes)

    image_upload = None
    if image_file is not None and image_bytes is not None:
        image_upload = (image_file.filename or "damage.png", image_file.content_type, image_bytes)

    package_metadata = persist_production_package(
        claim_payload=claim_payload,
        claim_json_bytes=claim_json_bytes,
        pdf_upload=pdf_upload,
        image_upload=image_upload,
        document_extraction_mode=extraction_result["document_extraction_mode"],
    )
    prediction = score_claims(claims=[claim_payload])[0]
    record = {
        "claim_payload": claim_payload,
        "structured_claim": claim_payload,
        "document_extraction": extraction_result,
        "package_metadata": package_metadata,
        "prediction": prediction,
        "policy": production_policy_metadata(),
    }
    record["record_path"] = persist_production_record(record)

    upsert_production_claim_intake(
        {
            **claim_payload,
            "document_extraction_mode": extraction_result["document_extraction_mode"],
            "chosen_sources": extraction_result.get("chosen_sources", {}),
            "mismatches": extraction_result.get("mismatches", []),
            "package_metadata": package_metadata,
            "structured_claim": claim_payload,
        },
        settings=app.state.postgres_settings,
    )

    upsert_production_claim_decision(
        {
            **claim_payload,
            "fraud_probability": prediction["fraud_probability"],
            "model_decision": prediction["model_decision"],
            "decision_threshold": prediction["decision_threshold"],
            "intake_mode": "json_plus_attachments",
            "package_directory": package_metadata["package_directory"],
            "record_path": record["record_path"],
            "chosen_sources": extraction_result.get("chosen_sources", {}),
            "mismatch_fields": [item["field"] for item in extraction_result.get("mismatches", [])],
            "package_metadata": package_metadata,
            "prediction_metadata": {
                "risk_flags": prediction["risk_flags"],
                "policy": production_policy_metadata(),
                "document_extraction_mode": extraction_result["document_extraction_mode"],
            },
        },
        settings=app.state.postgres_settings,
    )

    return {
        **prediction,
        "intake_mode": "json_plus_attachments",
        "structured_claim": claim_payload,
        "document_extraction": extraction_result,
        "package_metadata": package_metadata,
        "policy": production_policy_metadata(),
        "record_path": record["record_path"],
        "postgres_storage": {
            "persisted": True,
            "intake_table_name": app.state.postgres_settings.production_intake_table_name,
            "decision_table_name": app.state.postgres_settings.production_table_name,
        },
    }


@app.get("/health")
def healthcheck() -> JSONResponse:
    """Basic health check endpoint to verify that the API is running and can access the model and PostgreSQL."""
    return JSONResponse({"status": "ok", "policy": production_policy_metadata()})


@app.post("/predict")
def predict(payload: Any = Body(...)) -> Any:
    """Predict fraud probability and decision for a single claim or a batch of claims provided in the request body as JSON. The input can be either a single claim object or a list of claim objects. Each claim must contain the necessary fields for prediction. The response will include the fraud probability, model decision, and any inferred risk flags for each claim."""
    try:
        claims, is_batch = parse_predict_payload(payload)
        results = score_claims(claims=claims)
        return {"count": len(results), "items": results} if is_batch else results[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/package")
async def predict_package(
    claim_json: str | None = Form(default=None),
    claim_json_file: UploadFile | None = File(default=None),
    pdf_file: UploadFile | None = File(default=None),
    image_file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Ingest a claim package with structured claim data and attachments, extract and consolidate the information, score the claim using the trained model, and return the prediction along with metadata about the extraction and scoring process. This endpoint is designed for claims submitted with supporting documents, allowing for a more comprehensive evaluation compared to the /predict endpoint which accepts only structured data."""
    try:
        return await ingest_and_score_package(
            claim_json=claim_json,
            claim_json_file=claim_json_file,
            pdf_file=pdf_file,
            image_file=image_file,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/intake/package")
async def intake_package(
    claim_json: str | None = Form(default=None),
    claim_json_file: UploadFile | None = File(default=None),
    pdf_file: UploadFile | None = File(default=None),
    image_file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    """Ingest a claim package with structured claim data and attachments, extract and consolidate the information, score the claim using the trained model, store the intake and decision in PostgreSQL, and return the prediction along with metadata about the extraction, scoring, and storage process. This endpoint is intended for claims submitted through production channels, ensuring that all relevant information is captured, scored, and persisted for future reference and analysis."""
    try:
        return await ingest_and_score_package(
            claim_json=claim_json,
            claim_json_file=claim_json_file,
            pdf_file=pdf_file,
            image_file=image_file,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/claims")
def get_claims(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    location: str | None = Query(default=None),
    claim_type: str | None = Query(default=None),
    is_fraud: bool | None = Query(default=None),
    provider: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10_000),
) -> dict[str, Any]:
    """Query historical claims from PostgreSQL with optional filters for date range, location, claim type, fraud label, and provider. Returns a list of claims matching the criteria along with a count of the total results. This endpoint allows analysts to explore the historical claims data used for training and evaluation, enabling deeper insights into claim characteristics and model performance."""
    try:
        filters = query_filters(start_date, end_date, location, claim_type, is_fraud, provider)
        claims_df = fetch_historical_claims(
            filters=filters,
            limit=limit,
            settings=app.state.postgres_settings,
        )
        return {"count": int(len(claims_df)), "items": serialize_dataframe(claims_df)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/claims/export")
def export_claims(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    location: str | None = Query(default=None),
    claim_type: str | None = Query(default=None),
    is_fraud: bool | None = Query(default=None),
    provider: str | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=50_000),
) -> StreamingResponse:
    """Export historical claims from PostgreSQL as a CSV file with optional filters for date range, location, claim type, fraud label, and provider. Returns a streaming response that triggers a file download in the client's browser. This endpoint is designed for analysts who want to perform offline analysis or share subsets of the claims data, providing a convenient way to access and utilize the historical claims information outside of the API."""
    try:
        filters = query_filters(start_date, end_date, location, claim_type, is_fraud, provider)
        claims_df = fetch_historical_claims(
            filters=filters,
            limit=limit,
            settings=app.state.postgres_settings,
        )
        csv_buffer = StringIO()
        claims_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        headers = {
            "Content-Disposition": 'attachment; filename="claims_export.csv"',
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(iter([csv_buffer.getvalue()]), media_type="text/csv", headers=headers)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/counterfactual")
def counterfactual(request: CounterfactualRequest) -> dict[str, Any]:
    """Generate a counterfactual explanation for a given claim, either by providing the claim details directly in the request body or by referencing an existing claim through its claim_id. The endpoint will attempt to retrieve the claim information from PostgreSQL if only the claim_id is provided, and then use the trained model to generate a counterfactual explanation based on the claim's features and the specified decision threshold. The response will include the counterfactual explanation, metadata about the generation process, and relevant policy information."""
    try:
        if request.claim:
            claim_payload = ensure_claim_id(request.claim.model_dump(mode="json"))
        else:
            claim_payload = fetch_claim_by_id(
                claim_id=str(request.claim_id),
                include_production_intake=False,
                settings=app.state.postgres_settings,
            )
            if not claim_payload:
                production_record = None
                production_decision = fetch_production_decision_by_id(
                    claim_id=str(request.claim_id),
                    settings=app.state.postgres_settings,
                )
                if production_decision:
                    claim_payload = {
                        key: production_decision[key]
                        for key in (
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
                        )
                    }
                else:
                    production_record = load_production_record(str(request.claim_id))
                if not claim_payload and production_record:
                    claim_payload = production_record.get("structured_claim") or production_record.get("claim_payload")
                if not claim_payload:
                    production_intake = fetch_production_intake_by_id(
                        claim_id=str(request.claim_id),
                        settings=app.state.postgres_settings,
                    )
                    if production_intake:
                        claim_payload = {
                            key: production_intake[key]
                            for key in (
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
                            )
                        }
            if not claim_payload:
                raise HTTPException(status_code=404, detail="Claim not found.")
            for field in ("claim_date", "policy_start_date"):
                if isinstance(claim_payload.get(field), pd.Timestamp):
                    claim_payload[field] = claim_payload[field].date().isoformat()

        explanation = generate_counterfactual(
            claim_payload=claim_payload,
            model=app.state.model,
            metadata=app.state.metadata,
            decision_threshold=request.decision_threshold,
        )
        explanation["metadata"] = {"method": "heuristic_search", "model_type": "random_forest"}
        explanation["policy"] = production_policy_metadata()
        return explanation
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
