from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from src.config import DATA_SUMMARY_PATH, GROUND_TRUTH_TRAIN_PATH, PostgresSettings, SQL_VIEWS_SCRIPT_PATH
from src.db.postgres_connection import postgres_connection_context


CUSTOMERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    customer_id TEXT PRIMARY KEY,
    customer_age INTEGER NOT NULL,
    location TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

POLICIES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    policy_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL UNIQUE REFERENCES {customers_table_name} (customer_id) ON DELETE CASCADE,
    policy_start_date DATE NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CLAIMS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    claim_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES {customers_table_name} (customer_id) ON DELETE CASCADE,
    policy_id TEXT NOT NULL REFERENCES {policies_table_name} (policy_id) ON DELETE CASCADE,
    claim_date DATE NOT NULL,
    claim_amount NUMERIC(12, 2) NOT NULL,
    claim_type TEXT NOT NULL,
    num_previous_claims INTEGER NOT NULL,
    time_since_last_claim_days INTEGER NULL,
    service_provider_id TEXT NOT NULL,
    weather_condition TEXT NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PRODUCTION_INTAKE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    claim_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES {customers_table_name} (customer_id) ON DELETE CASCADE,
    policy_id TEXT NOT NULL REFERENCES {policies_table_name} (policy_id) ON DELETE CASCADE,
    claim_date DATE NOT NULL,
    policy_start_date DATE NOT NULL,
    policy_age_days INTEGER NOT NULL,
    claim_amount NUMERIC(12, 2) NOT NULL,
    claim_type TEXT NOT NULL,
    customer_age INTEGER NOT NULL,
    location TEXT NOT NULL,
    num_previous_claims INTEGER NOT NULL,
    time_since_last_claim_days INTEGER NULL,
    service_provider_id TEXT NOT NULL,
    weather_condition TEXT NOT NULL,
    document_extraction_mode TEXT NOT NULL,
    chosen_sources JSONB NOT NULL,
    mismatches JSONB NOT NULL,
    package_metadata JSONB NOT NULL,
    structured_claim JSONB NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PRODUCTION_DECISIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    claim_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES {customers_table_name} (customer_id) ON DELETE CASCADE,
    policy_id TEXT NOT NULL REFERENCES {policies_table_name} (policy_id) ON DELETE CASCADE,
    claim_date DATE NOT NULL,
    policy_start_date DATE NOT NULL,
    policy_age_days INTEGER NOT NULL,
    claim_amount NUMERIC(12, 2) NOT NULL,
    claim_type TEXT NOT NULL,
    customer_age INTEGER NOT NULL,
    location TEXT NOT NULL,
    num_previous_claims INTEGER NOT NULL,
    time_since_last_claim_days INTEGER NULL,
    service_provider_id TEXT NOT NULL,
    weather_condition TEXT NOT NULL,
    fraud_probability DOUBLE PRECISION NOT NULL,
    model_decision TEXT NOT NULL,
    decision_threshold DOUBLE PRECISION NOT NULL,
    intake_mode TEXT NOT NULL,
    package_directory TEXT NULL,
    record_path TEXT NULL,
    chosen_sources JSONB NULL,
    mismatch_fields JSONB NULL,
    package_metadata JSONB NULL,
    prediction_metadata JSONB NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CUSTOMER_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_customers_location ON {table_name} (location);",
]

POLICY_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_policies_customer_id ON {table_name} (customer_id);",
    "CREATE INDEX IF NOT EXISTS idx_policies_start_date ON {table_name} (policy_start_date);",
]

CLAIMS_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_claims_claim_date ON {table_name} (claim_date);",
    "CREATE INDEX IF NOT EXISTS idx_claims_service_provider ON {table_name} (service_provider_id);",
    "CREATE INDEX IF NOT EXISTS idx_claims_claim_type ON {table_name} (claim_type);",
    "CREATE INDEX IF NOT EXISTS idx_claims_customer_id ON {table_name} (customer_id);",
]

PRODUCTION_INTAKE_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_prod_intake_claim_date ON {table_name} (claim_date);",
    "CREATE INDEX IF NOT EXISTS idx_prod_intake_service_provider ON {table_name} (service_provider_id);",
    "CREATE INDEX IF NOT EXISTS idx_prod_intake_location ON {table_name} (location);",
]

PRODUCTION_DECISIONS_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_prod_decisions_claim_date ON {table_name} (claim_date);",
    "CREATE INDEX IF NOT EXISTS idx_prod_decisions_decision ON {table_name} (model_decision);",
    "CREATE INDEX IF NOT EXISTS idx_prod_decisions_service_provider ON {table_name} (service_provider_id);",
]

CUSTOMERS_UPSERT_SQL = """
INSERT INTO {table_name} (
    customer_id,
    customer_age,
    location
) VALUES %s
ON CONFLICT (customer_id) DO UPDATE SET
    customer_age = EXCLUDED.customer_age,
    location = EXCLUDED.location,
    updated_at = CURRENT_TIMESTAMP;
"""

POLICIES_UPSERT_SQL = """
INSERT INTO {table_name} (
    policy_id,
    customer_id,
    policy_start_date
) VALUES %s
ON CONFLICT (customer_id) DO UPDATE SET
    policy_start_date = EXCLUDED.policy_start_date,
    updated_at = CURRENT_TIMESTAMP;
"""

CLAIMS_UPSERT_SQL = """
INSERT INTO {table_name} (
    claim_id,
    customer_id,
    policy_id,
    claim_date,
    claim_amount,
    claim_type,
    num_previous_claims,
    time_since_last_claim_days,
    service_provider_id,
    weather_condition,
    is_fraud
) VALUES %s
ON CONFLICT (claim_id) DO UPDATE SET
    customer_id = EXCLUDED.customer_id,
    policy_id = EXCLUDED.policy_id,
    claim_date = EXCLUDED.claim_date,
    claim_amount = EXCLUDED.claim_amount,
    claim_type = EXCLUDED.claim_type,
    num_previous_claims = EXCLUDED.num_previous_claims,
    time_since_last_claim_days = EXCLUDED.time_since_last_claim_days,
    service_provider_id = EXCLUDED.service_provider_id,
    weather_condition = EXCLUDED.weather_condition,
    is_fraud = EXCLUDED.is_fraud,
    updated_at = CURRENT_TIMESTAMP;
"""

PRODUCTION_INTAKE_UPSERT_SQL = """
INSERT INTO {table_name} (
    claim_id,
    customer_id,
    policy_id,
    claim_date,
    policy_start_date,
    policy_age_days,
    claim_amount,
    claim_type,
    customer_age,
    location,
    num_previous_claims,
    time_since_last_claim_days,
    service_provider_id,
    weather_condition,
    document_extraction_mode,
    chosen_sources,
    mismatches,
    package_metadata,
    structured_claim
) VALUES %s
ON CONFLICT (claim_id) DO UPDATE SET
    customer_id = EXCLUDED.customer_id,
    policy_id = EXCLUDED.policy_id,
    claim_date = EXCLUDED.claim_date,
    policy_start_date = EXCLUDED.policy_start_date,
    policy_age_days = EXCLUDED.policy_age_days,
    claim_amount = EXCLUDED.claim_amount,
    claim_type = EXCLUDED.claim_type,
    customer_age = EXCLUDED.customer_age,
    location = EXCLUDED.location,
    num_previous_claims = EXCLUDED.num_previous_claims,
    time_since_last_claim_days = EXCLUDED.time_since_last_claim_days,
    service_provider_id = EXCLUDED.service_provider_id,
    weather_condition = EXCLUDED.weather_condition,
    document_extraction_mode = EXCLUDED.document_extraction_mode,
    chosen_sources = EXCLUDED.chosen_sources,
    mismatches = EXCLUDED.mismatches,
    package_metadata = EXCLUDED.package_metadata,
    structured_claim = EXCLUDED.structured_claim,
    updated_at = CURRENT_TIMESTAMP;
"""

PRODUCTION_DECISIONS_UPSERT_SQL = """
INSERT INTO {table_name} (
    claim_id,
    customer_id,
    policy_id,
    claim_date,
    policy_start_date,
    policy_age_days,
    claim_amount,
    claim_type,
    customer_age,
    location,
    num_previous_claims,
    time_since_last_claim_days,
    service_provider_id,
    weather_condition,
    fraud_probability,
    model_decision,
    decision_threshold,
    intake_mode,
    package_directory,
    record_path,
    chosen_sources,
    mismatch_fields,
    package_metadata,
    prediction_metadata
) VALUES %s
ON CONFLICT (claim_id) DO UPDATE SET
    customer_id = EXCLUDED.customer_id,
    policy_id = EXCLUDED.policy_id,
    claim_date = EXCLUDED.claim_date,
    policy_start_date = EXCLUDED.policy_start_date,
    policy_age_days = EXCLUDED.policy_age_days,
    claim_amount = EXCLUDED.claim_amount,
    claim_type = EXCLUDED.claim_type,
    customer_age = EXCLUDED.customer_age,
    location = EXCLUDED.location,
    num_previous_claims = EXCLUDED.num_previous_claims,
    time_since_last_claim_days = EXCLUDED.time_since_last_claim_days,
    service_provider_id = EXCLUDED.service_provider_id,
    weather_condition = EXCLUDED.weather_condition,
    fraud_probability = EXCLUDED.fraud_probability,
    model_decision = EXCLUDED.model_decision,
    decision_threshold = EXCLUDED.decision_threshold,
    intake_mode = EXCLUDED.intake_mode,
    package_directory = EXCLUDED.package_directory,
    record_path = EXCLUDED.record_path,
    chosen_sources = EXCLUDED.chosen_sources,
    mismatch_fields = EXCLUDED.mismatch_fields,
    package_metadata = EXCLUDED.package_metadata,
    prediction_metadata = EXCLUDED.prediction_metadata,
    updated_at = CURRENT_TIMESTAMP;
"""


def derive_policy_id(customer_id: str) -> str:
    return f"POL-{customer_id}"


def load_historical_claims_dataframe(input_path: Path = GROUND_TRUTH_TRAIN_PATH) -> pd.DataFrame:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    dataframe = pd.DataFrame.from_records(payload)
    if dataframe.empty:
        raise ValueError(f"No historical claims found in {input_path}.")

    for column in ("claim_date", "policy_start_date"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(dataframe[column])
    for column in ("claim_amount", "customer_age", "num_previous_claims", "time_since_last_claim_days", "is_fraud"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe["policy_id"] = dataframe["customer_id"].astype(str).map(derive_policy_id)
    return dataframe


def normalize_claim_dataframe(dataframe: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    if isinstance(dataframe, dict):
        dataframe = pd.DataFrame([dataframe])
    elif isinstance(dataframe, list):
        dataframe = pd.DataFrame(dataframe)
    else:
        dataframe = dataframe.copy()

    required_columns = [
        "claim_id",
        "customer_id",
        "claim_date",
        "policy_start_date",
        "claim_amount",
        "claim_type",
        "customer_age",
        "num_previous_claims",
        "time_since_last_claim_days",
        "service_provider_id",
        "location",
        "weather_condition",
    ]
    for column in required_columns:
        if column not in dataframe.columns:
            dataframe[column] = pd.NA

    dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"], errors="coerce")
    dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"], errors="coerce")
    for column in ("claim_amount", "customer_age", "num_previous_claims", "time_since_last_claim_days", "policy_age_days"):
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
        else:
            dataframe[column] = pd.NA

    if dataframe["policy_age_days"].isna().all():
        dataframe["policy_age_days"] = (dataframe["claim_date"] - dataframe["policy_start_date"]).dt.days

    dataframe["policy_id"] = dataframe["customer_id"].astype(str).map(derive_policy_id)
    dataframe["claim_type"] = dataframe["claim_type"].astype("string")
    dataframe["service_provider_id"] = dataframe["service_provider_id"].astype("string")
    dataframe["location"] = dataframe["location"].astype("string")
    dataframe["weather_condition"] = dataframe["weather_condition"].astype("string")
    return dataframe


def build_customer_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    customer_rows = dataframe.loc[:, ["customer_id", "customer_age", "location"]].copy()
    customer_rows = customer_rows.dropna(subset=["customer_id"]).drop_duplicates("customer_id")
    customer_rows["customer_age"] = pd.to_numeric(customer_rows["customer_age"], errors="coerce").astype(int)
    customer_rows["location"] = customer_rows["location"].astype(str)
    return customer_rows


def build_policy_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    policy_rows = dataframe.loc[:, ["policy_id", "customer_id", "policy_start_date"]].copy()
    policy_rows = policy_rows.dropna(subset=["policy_id", "customer_id"]).drop_duplicates("customer_id")
    policy_rows["policy_start_date"] = pd.to_datetime(policy_rows["policy_start_date"]).dt.date
    return policy_rows


def build_claim_rows(dataframe: pd.DataFrame, include_label: bool = True) -> pd.DataFrame:
    columns = [
        "claim_id",
        "customer_id",
        "policy_id",
        "claim_date",
        "claim_amount",
        "claim_type",
        "num_previous_claims",
        "time_since_last_claim_days",
        "service_provider_id",
        "weather_condition",
    ]
    if include_label:
        columns.append("is_fraud")

    claim_rows = dataframe.loc[:, columns].copy()
    claim_rows["claim_date"] = pd.to_datetime(claim_rows["claim_date"]).dt.date
    claim_rows["claim_amount"] = pd.to_numeric(claim_rows["claim_amount"], errors="coerce")
    claim_rows["num_previous_claims"] = pd.to_numeric(claim_rows["num_previous_claims"], errors="coerce").astype(int)
    claim_rows["time_since_last_claim_days"] = claim_rows["time_since_last_claim_days"].apply(
        lambda value: None if pd.isna(value) else int(value)
    )
    if include_label:
        claim_rows["is_fraud"] = pd.to_numeric(claim_rows["is_fraud"], errors="coerce").astype(int)
    return claim_rows


def create_customers_table(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(CUSTOMERS_TABLE_SQL).format(
                    table_name=sql.Identifier(postgres_settings.customers_table_name)
                )
            )
            for statement in CUSTOMER_INDEX_STATEMENTS:
                cursor.execute(sql.SQL(statement).format(table_name=sql.Identifier(postgres_settings.customers_table_name)))


def create_policies_table(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(POLICIES_TABLE_SQL).format(
                    table_name=sql.Identifier(postgres_settings.policies_table_name),
                    customers_table_name=sql.Identifier(postgres_settings.customers_table_name),
                )
            )
            for statement in POLICY_INDEX_STATEMENTS:
                cursor.execute(sql.SQL(statement).format(table_name=sql.Identifier(postgres_settings.policies_table_name)))


def create_historical_claims_table(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(CLAIMS_TABLE_SQL).format(
                    table_name=sql.Identifier(postgres_settings.table_name),
                    customers_table_name=sql.Identifier(postgres_settings.customers_table_name),
                    policies_table_name=sql.Identifier(postgres_settings.policies_table_name),
                )
            )
            for statement in CLAIMS_INDEX_STATEMENTS:
                cursor.execute(sql.SQL(statement).format(table_name=sql.Identifier(postgres_settings.table_name)))


def create_production_intake_table(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(PRODUCTION_INTAKE_TABLE_SQL).format(
                    table_name=sql.Identifier(postgres_settings.production_intake_table_name),
                    customers_table_name=sql.Identifier(postgres_settings.customers_table_name),
                    policies_table_name=sql.Identifier(postgres_settings.policies_table_name),
                )
            )
            for statement in PRODUCTION_INTAKE_INDEX_STATEMENTS:
                cursor.execute(
                    sql.SQL(statement).format(table_name=sql.Identifier(postgres_settings.production_intake_table_name))
                )


def create_production_decisions_table(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(PRODUCTION_DECISIONS_TABLE_SQL).format(
                    table_name=sql.Identifier(postgres_settings.production_table_name),
                    customers_table_name=sql.Identifier(postgres_settings.customers_table_name),
                    policies_table_name=sql.Identifier(postgres_settings.policies_table_name),
                )
            )
            for statement in PRODUCTION_DECISIONS_INDEX_STATEMENTS:
                cursor.execute(
                    sql.SQL(statement).format(table_name=sql.Identifier(postgres_settings.production_table_name))
                )


def create_analytics_views(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    script = SQL_VIEWS_SCRIPT_PATH.read_text(encoding="utf-8")
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(script)


def create_supporting_tables(settings: PostgresSettings | None = None) -> None:
    create_customers_table(settings=settings)
    create_policies_table(settings=settings)
    create_historical_claims_table(settings=settings)
    create_production_intake_table(settings=settings)
    create_production_decisions_table(settings=settings)
    create_analytics_views(settings=settings)


def drop_supporting_tables(settings: PostgresSettings | None = None) -> None:
    postgres_settings = settings or PostgresSettings.from_env()
    drop_statement = sql.SQL(
        """
        DROP TABLE IF EXISTS
            {production_decisions},
            {production_intake},
            {claims},
            {policies},
            {customers}
        CASCADE;
        """
    ).format(
        production_decisions=sql.Identifier(postgres_settings.production_table_name),
        production_intake=sql.Identifier(postgres_settings.production_intake_table_name),
        claims=sql.Identifier(postgres_settings.table_name),
        policies=sql.Identifier(postgres_settings.policies_table_name),
        customers=sql.Identifier(postgres_settings.customers_table_name),
    )
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute(drop_statement)


def rebuild_supporting_tables(settings: PostgresSettings | None = None) -> None:
    drop_supporting_tables(settings=settings)
    create_supporting_tables(settings=settings)


def upsert_customers(dataframe: pd.DataFrame, settings: PostgresSettings | None = None, page_size: int = 5_000) -> int:
    postgres_settings = settings or PostgresSettings.from_env()
    customer_rows = build_customer_rows(dataframe)
    records = [
        (
            record["customer_id"],
            int(record["customer_age"]),
            str(record["location"]),
        )
        for record in customer_rows.to_dict(orient="records")
    ]
    upsert_query = sql.SQL(CUSTOMERS_UPSERT_SQL).format(table_name=sql.Identifier(postgres_settings.customers_table_name))
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), records, page_size=page_size)
    return len(records)


def upsert_policies(dataframe: pd.DataFrame, settings: PostgresSettings | None = None, page_size: int = 5_000) -> int:
    postgres_settings = settings or PostgresSettings.from_env()
    policy_rows = build_policy_rows(dataframe)
    records = [
        (
            record["policy_id"],
            record["customer_id"],
            record["policy_start_date"],
        )
        for record in policy_rows.to_dict(orient="records")
    ]
    upsert_query = sql.SQL(POLICIES_UPSERT_SQL).format(table_name=sql.Identifier(postgres_settings.policies_table_name))
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), records, page_size=page_size)
    return len(records)


def upsert_historical_claims(
    dataframe: pd.DataFrame,
    settings: PostgresSettings | None = None,
    page_size: int = 5_000,
) -> int:
    postgres_settings = settings or PostgresSettings.from_env()
    normalized_df = normalize_claim_dataframe(dataframe)
    upsert_customers(normalized_df, settings=postgres_settings, page_size=page_size)
    upsert_policies(normalized_df, settings=postgres_settings, page_size=page_size)

    claim_rows = build_claim_rows(normalized_df, include_label=True)
    records = [
        (
            record["claim_id"],
            record["customer_id"],
            record["policy_id"],
            record["claim_date"],
            float(record["claim_amount"]),
            record["claim_type"],
            int(record["num_previous_claims"]),
            None if pd.isna(record["time_since_last_claim_days"]) else int(record["time_since_last_claim_days"]),
            record["service_provider_id"],
            record["weather_condition"],
            bool(record["is_fraud"]),
        )
        for record in claim_rows.to_dict(orient="records")
    ]
    upsert_query = sql.SQL(CLAIMS_UPSERT_SQL).format(table_name=sql.Identifier(postgres_settings.table_name))
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), records, page_size=page_size)
    return len(records)


def upsert_production_claim_intake(
    record: dict[str, Any],
    settings: PostgresSettings | None = None,
) -> str:
    postgres_settings = settings or PostgresSettings.from_env()
    normalized_df = normalize_claim_dataframe(record)
    upsert_customers(normalized_df, settings=postgres_settings, page_size=1)
    upsert_policies(normalized_df, settings=postgres_settings, page_size=1)
    normalized_record = normalized_df.iloc[0].to_dict()
    payload = (
        normalized_record["claim_id"],
        normalized_record["customer_id"],
        normalized_record["policy_id"],
        pd.to_datetime(normalized_record["claim_date"]).date(),
        pd.to_datetime(normalized_record["policy_start_date"]).date(),
        int(normalized_record["policy_age_days"]),
        float(normalized_record["claim_amount"]),
        normalized_record["claim_type"],
        int(normalized_record["customer_age"]),
        normalized_record["location"],
        int(normalized_record["num_previous_claims"]),
        None if pd.isna(normalized_record.get("time_since_last_claim_days")) else int(normalized_record["time_since_last_claim_days"]),
        normalized_record["service_provider_id"],
        normalized_record["weather_condition"],
        normalized_record.get("document_extraction_mode", "implemented_rule_based_pdf_and_image_extraction"),
        Json(normalized_record.get("chosen_sources", {})),
        Json(normalized_record.get("mismatches", [])),
        Json(normalized_record.get("package_metadata", {})),
        Json(normalized_record.get("structured_claim", {})),
    )
    upsert_query = sql.SQL(PRODUCTION_INTAKE_UPSERT_SQL).format(
        table_name=sql.Identifier(postgres_settings.production_intake_table_name)
    )
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), [payload], page_size=1)
    return str(normalized_record["claim_id"])


def upsert_production_claim_intakes(
    dataframe: pd.DataFrame,
    settings: PostgresSettings | None = None,
    page_size: int = 5_000,
) -> int:
    postgres_settings = settings or PostgresSettings.from_env()
    normalized_df = normalize_claim_dataframe(dataframe)
    upsert_customers(normalized_df, settings=postgres_settings, page_size=page_size)
    upsert_policies(normalized_df, settings=postgres_settings, page_size=page_size)

    records = []
    for record in normalized_df.to_dict(orient="records"):
        records.append(
            (
                record["claim_id"],
                record["customer_id"],
                record["policy_id"],
                pd.to_datetime(record["claim_date"]).date(),
                pd.to_datetime(record["policy_start_date"]).date(),
                int(record["policy_age_days"]),
                float(record["claim_amount"]),
                record["claim_type"],
                int(record["customer_age"]),
                record["location"],
                int(record["num_previous_claims"]),
                None if pd.isna(record.get("time_since_last_claim_days")) else int(record["time_since_last_claim_days"]),
                record["service_provider_id"],
                record["weather_condition"],
                record.get("document_extraction_mode", "implemented_rule_based_pdf_and_image_extraction"),
                Json(record.get("chosen_sources", {})),
                Json(record.get("mismatches", [])),
                Json(record.get("package_metadata", {})),
                Json(record.get("structured_claim", {})),
            )
        )
    upsert_query = sql.SQL(PRODUCTION_INTAKE_UPSERT_SQL).format(
        table_name=sql.Identifier(postgres_settings.production_intake_table_name)
    )
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), records, page_size=page_size)
    return len(records)


def upsert_production_claim_decision(
    record: dict[str, Any],
    settings: PostgresSettings | None = None,
) -> str:
    postgres_settings = settings or PostgresSettings.from_env()
    normalized_df = normalize_claim_dataframe(record)
    upsert_customers(normalized_df, settings=postgres_settings, page_size=1)
    upsert_policies(normalized_df, settings=postgres_settings, page_size=1)
    normalized_record = normalized_df.iloc[0].to_dict()
    payload = (
        normalized_record["claim_id"],
        normalized_record["customer_id"],
        normalized_record["policy_id"],
        pd.to_datetime(normalized_record["claim_date"]).date(),
        pd.to_datetime(normalized_record["policy_start_date"]).date(),
        int(normalized_record["policy_age_days"]),
        float(normalized_record["claim_amount"]),
        normalized_record["claim_type"],
        int(normalized_record["customer_age"]),
        normalized_record["location"],
        int(normalized_record["num_previous_claims"]),
        None if pd.isna(normalized_record.get("time_since_last_claim_days")) else int(normalized_record["time_since_last_claim_days"]),
        normalized_record["service_provider_id"],
        normalized_record["weather_condition"],
        float(normalized_record["fraud_probability"]),
        normalized_record["model_decision"],
        float(normalized_record["decision_threshold"]),
        normalized_record["intake_mode"],
        normalized_record.get("package_directory"),
        normalized_record.get("record_path"),
        Json(normalized_record.get("chosen_sources", {})),
        Json(normalized_record.get("mismatch_fields", [])),
        Json(normalized_record.get("package_metadata", {})),
        Json(normalized_record.get("prediction_metadata", {})),
    )
    upsert_query = sql.SQL(PRODUCTION_DECISIONS_UPSERT_SQL).format(
        table_name=sql.Identifier(postgres_settings.production_table_name)
    )
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), [payload], page_size=1)
    return str(normalized_record["claim_id"])


def upsert_production_claim_decisions(
    dataframe: pd.DataFrame,
    settings: PostgresSettings | None = None,
    page_size: int = 5_000,
) -> int:
    postgres_settings = settings or PostgresSettings.from_env()
    normalized_df = normalize_claim_dataframe(dataframe)
    upsert_customers(normalized_df, settings=postgres_settings, page_size=page_size)
    upsert_policies(normalized_df, settings=postgres_settings, page_size=page_size)

    records = []
    for record in normalized_df.to_dict(orient="records"):
        records.append(
            (
                record["claim_id"],
                record["customer_id"],
                record["policy_id"],
                pd.to_datetime(record["claim_date"]).date(),
                pd.to_datetime(record["policy_start_date"]).date(),
                int(record["policy_age_days"]),
                float(record["claim_amount"]),
                record["claim_type"],
                int(record["customer_age"]),
                record["location"],
                int(record["num_previous_claims"]),
                None if pd.isna(record.get("time_since_last_claim_days")) else int(record["time_since_last_claim_days"]),
                record["service_provider_id"],
                record["weather_condition"],
                float(record["fraud_probability"]),
                record["model_decision"],
                float(record["decision_threshold"]),
                record["intake_mode"],
                record.get("package_directory"),
                record.get("record_path"),
                Json(record.get("chosen_sources", {})),
                Json(record.get("mismatch_fields", [])),
                Json(record.get("package_metadata", {})),
                Json(record.get("prediction_metadata", {})),
            )
        )
    upsert_query = sql.SQL(PRODUCTION_DECISIONS_UPSERT_SQL).format(
        table_name=sql.Identifier(postgres_settings.production_table_name)
    )
    with postgres_connection_context(settings=postgres_settings) as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, upsert_query.as_string(connection), records, page_size=page_size)
    return len(records)


def main(input_path: Path = GROUND_TRUTH_TRAIN_PATH, reset_db: bool = False) -> dict[str, object]:
    settings = PostgresSettings.from_env()
    if reset_db:
        rebuild_supporting_tables(settings=settings)
    else:
        create_supporting_tables(settings=settings)

    historical_df = load_historical_claims_dataframe(input_path=input_path)
    loaded_rows = upsert_historical_claims(dataframe=historical_df, settings=settings)

    summary = {
        "customers_table_name": settings.customers_table_name,
        "policies_table_name": settings.policies_table_name,
        "historical_table_name": settings.table_name,
        "production_intake_table_name": settings.production_intake_table_name,
        "production_decisions_table_name": settings.production_table_name,
        "rows_loaded": loaded_rows,
        "input_path": str(input_path),
        "input_format": input_path.suffix.lstrip(".").lower(),
        "historical_date_min": historical_df["claim_date"].min().date().isoformat(),
        "historical_date_max": historical_df["claim_date"].max().date().isoformat(),
        "structured_source_stage": "client_package_extraction",
        "database_schema": "normalized_5_table_portfolio_schema",
    }

    if DATA_SUMMARY_PATH.exists():
        existing = json.loads(DATA_SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        existing = {}
    existing["postgres_load_summary"] = summary
    DATA_SUMMARY_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load historical extracted claims into PostgreSQL.")
    parser.add_argument("--input-path", type=Path, default=GROUND_TRUTH_TRAIN_PATH)
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()
    main(input_path=args.input_path, reset_db=args.reset_db)
