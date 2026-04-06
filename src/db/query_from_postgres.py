from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
from psycopg2 import sql

from src.config import PostgresSettings, SQL_ANALYST_REPORT_PATH
from src.db.postgres_connection import postgres_connection_context


def build_filter_clause(filters: dict[str, Any] | None = None, *, fraud_column: str | None) -> tuple[sql.Composed, list[Any]]:
    filters = filters or {}
    clauses: list[sql.SQL] = []
    params: list[Any] = []

    if filters.get("start_date"):
        clauses.append(sql.SQL("claim_date >= %s"))
        params.append(filters["start_date"])
    if filters.get("end_date"):
        clauses.append(sql.SQL("claim_date <= %s"))
        params.append(filters["end_date"])
    if filters.get("location"):
        clauses.append(sql.SQL("location = %s"))
        params.append(filters["location"])
    if filters.get("claim_type"):
        clauses.append(sql.SQL("claim_type = %s"))
        params.append(filters["claim_type"])
    if filters.get("provider"):
        clauses.append(sql.SQL("service_provider_id = %s"))
        params.append(filters["provider"])
    if filters.get("is_fraud") is not None and fraud_column is not None:
        clauses.append(sql.SQL("{} = %s").format(sql.Identifier(fraud_column)))
        params.append(bool(filters["is_fraud"]))

    if not clauses:
        return sql.SQL(""), params
    return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses), params


def historical_view_identifier(postgres_settings: PostgresSettings) -> sql.Identifier:
    return sql.Identifier(postgres_settings.historical_view_name)


def production_intake_view_identifier(postgres_settings: PostgresSettings) -> sql.Identifier:
    return sql.Identifier(postgres_settings.production_intake_view_name)


def production_decisions_view_identifier(postgres_settings: PostgresSettings) -> sql.Identifier:
    return sql.Identifier(postgres_settings.production_view_name)


def fetch_historical_claims(
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    postgres_settings = settings or PostgresSettings.from_env()
    where_clause, params = build_filter_clause(filters=filters, fraud_column="is_fraud")
    limit_clause = sql.SQL("")
    if limit:
        limit_clause = sql.SQL(" LIMIT %s")
        params.append(int(limit))

    query = (
        sql.SQL(
            """
            SELECT
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
                is_fraud
            FROM {view_name}
            """
        ).format(view_name=historical_view_identifier(postgres_settings))
        + where_clause
        + sql.SQL(" ORDER BY claim_date ASC, claim_id ASC")
        + limit_clause
    )

    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"])
        dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"])
    return dataframe


def fetch_production_decisions(
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    postgres_settings = settings or PostgresSettings.from_env()
    where_clause, params = build_filter_clause(filters=filters, fraud_column=None)
    filters = filters or {}
    if filters.get("is_fraud") is not None:
        decision_clause = sql.SQL("model_decision = %s")
        decision_value = "fraud" if bool(filters["is_fraud"]) else "non_fraud"
        if params:
            where_clause = where_clause + sql.SQL(" AND ") + decision_clause
        else:
            where_clause = sql.SQL(" WHERE ") + decision_clause
        params.append(decision_value)
    limit_clause = sql.SQL("")
    if limit:
        limit_clause = sql.SQL(" LIMIT %s")
        params.append(int(limit))

    query = (
        sql.SQL(
            """
            SELECT
                claim_id,
                customer_id,
                policy_id,
                claim_date,
                policy_start_date,
                policy_age_days,
                claim_amount,
                claim_type,
                customer_age,
                num_previous_claims,
                time_since_last_claim_days,
                service_provider_id,
                location,
                weather_condition,
                fraud_probability,
                model_decision,
                decision_threshold,
                intake_mode,
                package_directory,
                record_path,
                created_at,
                updated_at
            FROM {view_name}
            """
        ).format(view_name=production_decisions_view_identifier(postgres_settings))
        + where_clause
        + sql.SQL(" ORDER BY claim_date DESC, updated_at DESC")
        + limit_clause
    )

    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"])
        dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"])
    return dataframe


def fetch_production_intake_claims(
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    settings: PostgresSettings | None = None,
) -> pd.DataFrame:
    postgres_settings = settings or PostgresSettings.from_env()
    where_clause, params = build_filter_clause(filters=filters, fraud_column=None)
    limit_clause = sql.SQL("")
    if limit:
        limit_clause = sql.SQL(" LIMIT %s")
        params.append(int(limit))

    query = (
        sql.SQL(
            """
            SELECT
                claim_id,
                customer_id,
                policy_id,
                claim_date,
                policy_start_date,
                policy_age_days,
                claim_amount,
                claim_type,
                customer_age,
                num_previous_claims,
                time_since_last_claim_days,
                service_provider_id,
                location,
                weather_condition,
                document_extraction_mode,
                chosen_sources,
                mismatches,
                package_metadata,
                structured_claim,
                created_at,
                updated_at
            FROM {view_name}
            """
        ).format(view_name=production_intake_view_identifier(postgres_settings))
        + where_clause
        + sql.SQL(" ORDER BY claim_date DESC, updated_at DESC")
        + limit_clause
    )

    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe["claim_date"] = pd.to_datetime(dataframe["claim_date"])
        dataframe["policy_start_date"] = pd.to_datetime(dataframe["policy_start_date"])
    return dataframe


def fetch_claim_by_id(
    claim_id: str,
    include_production_intake: bool = False,
    settings: PostgresSettings | None = None,
) -> dict[str, Any] | None:
    postgres_settings = settings or PostgresSettings.from_env()
    query = sql.SQL(
        """
        SELECT
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
            is_fraud
        FROM {view_name}
        WHERE claim_id = %s
        LIMIT 1
        """
    ).format(view_name=historical_view_identifier(postgres_settings))
    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, [claim_id])
            historical_match = cursor.fetchone()
    if historical_match:
        return dict(historical_match)

    if include_production_intake:
        intake_match = fetch_production_intake_by_id(claim_id=claim_id, settings=postgres_settings)
        if intake_match:
            return intake_match

    return None


def fetch_production_intake_by_id(
    claim_id: str,
    settings: PostgresSettings | None = None,
) -> dict[str, Any] | None:
    postgres_settings = settings or PostgresSettings.from_env()
    query = sql.SQL(
        """
        SELECT
            claim_id,
            customer_id,
            policy_id,
            claim_date,
            policy_start_date,
            policy_age_days,
            claim_amount,
            claim_type,
            customer_age,
            num_previous_claims,
            time_since_last_claim_days,
            service_provider_id,
            location,
            weather_condition,
            document_extraction_mode,
            chosen_sources,
            mismatches,
            package_metadata,
            structured_claim
        FROM {view_name}
        WHERE claim_id = %s
        LIMIT 1
        """
    ).format(view_name=production_intake_view_identifier(postgres_settings))
    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, [claim_id])
            match = cursor.fetchone()
    return None if match is None else dict(match)


def fetch_production_decision_by_id(
    claim_id: str,
    settings: PostgresSettings | None = None,
) -> dict[str, Any] | None:
    postgres_settings = settings or PostgresSettings.from_env()
    query = sql.SQL(
        """
        SELECT
            claim_id,
            customer_id,
            policy_id,
            claim_date,
            policy_start_date,
            policy_age_days,
            claim_amount,
            claim_type,
            customer_age,
            num_previous_claims,
            time_since_last_claim_days,
            service_provider_id,
            location,
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
        FROM {view_name}
        WHERE claim_id = %s
        LIMIT 1
        """
    ).format(view_name=production_decisions_view_identifier(postgres_settings))
    with postgres_connection_context(settings=postgres_settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, [claim_id])
            match = cursor.fetchone()
    return None if match is None else dict(match)


def table_has_rows(table_name: str, settings: PostgresSettings) -> bool:
    query = sql.SQL("SELECT EXISTS (SELECT 1 FROM {table_name} LIMIT 1) AS has_rows").format(
        table_name=sql.Identifier(table_name)
    )
    with postgres_connection_context(settings=settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
    return bool(result and result["has_rows"])


def fetch_rows_as_dicts(query: sql.Composed, params: list[Any], settings: PostgresSettings) -> list[dict[str, Any]]:
    with postgres_connection_context(settings=settings, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    return [dict(row) for row in rows]


def build_sql_analyst_report(settings: PostgresSettings | None = None) -> dict[str, Any]:
    postgres_settings = settings or PostgresSettings.from_env()

    overview_query = sql.SQL(
        """
        SELECT
            COUNT(*) AS historical_claims,
            ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END)::numeric, 4) AS fraud_rate,
            ROUND(AVG(customer_age)::numeric, 2) AS average_customer_age,
            ROUND(AVG(policy_age_days)::numeric, 2) AS average_policy_age_days,
            MIN(claim_date) AS min_claim_date,
            MAX(claim_date) AS max_claim_date
        FROM {view_name}
        """
    ).format(view_name=historical_view_identifier(postgres_settings))
    fraud_by_claim_type_query = sql.SQL(
        """
        SELECT
            claim_type,
            COUNT(*) AS claim_count,
            ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END)::numeric, 4) AS fraud_rate,
            ROUND(AVG(claim_amount)::numeric, 2) AS average_claim_amount
        FROM {view_name}
        GROUP BY claim_type
        ORDER BY fraud_rate DESC, claim_count DESC
        """
    ).format(view_name=historical_view_identifier(postgres_settings))
    suspicious_providers_query = sql.SQL(
        """
        SELECT
            service_provider_id,
            COUNT(*) AS claim_count,
            ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END)::numeric, 4) AS fraud_rate,
            ROUND(AVG(claim_amount)::numeric, 2) AS average_claim_amount
        FROM {view_name}
        GROUP BY service_provider_id
        HAVING COUNT(*) >= 25
        ORDER BY fraud_rate DESC, claim_count DESC
        LIMIT 10
        """
    ).format(view_name=historical_view_identifier(postgres_settings))
    monthly_trend_query = sql.SQL(
        """
        SELECT
            DATE_TRUNC('month', claim_date)::date AS claim_month,
            COUNT(*) AS claim_count,
            ROUND(AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END)::numeric, 4) AS fraud_rate
        FROM {view_name}
        GROUP BY DATE_TRUNC('month', claim_date)
        ORDER BY claim_month ASC
        """
    ).format(view_name=historical_view_identifier(postgres_settings))

    report = {
        "overview": fetch_rows_as_dicts(overview_query, [], postgres_settings)[0],
        "fraud_by_claim_type": fetch_rows_as_dicts(fraud_by_claim_type_query, [], postgres_settings),
        "suspicious_providers": fetch_rows_as_dicts(suspicious_providers_query, [], postgres_settings),
        "monthly_trend": fetch_rows_as_dicts(monthly_trend_query, [], postgres_settings),
    }

    if table_has_rows(postgres_settings.production_table_name, postgres_settings):
        production_query = sql.SQL(
            """
            SELECT
                COUNT(*) AS production_claims_scored,
                ROUND(AVG(fraud_probability)::numeric, 4) AS average_fraud_probability,
                ROUND(AVG(CASE WHEN model_decision = 'fraud' THEN 1.0 ELSE 0.0 END)::numeric, 4) AS predicted_fraud_rate,
                MIN(claim_date) AS min_claim_date,
                MAX(claim_date) AS max_claim_date
            FROM {view_name}
            """
        ).format(view_name=production_decisions_view_identifier(postgres_settings))
        report["production_decisions_overview"] = fetch_rows_as_dicts(production_query, [], postgres_settings)[0]
    else:
        report["production_decisions_overview"] = {
            "production_claims_scored": 0,
            "note": "No production decisions stored in PostgreSQL yet.",
        }

    if table_has_rows(postgres_settings.production_intake_table_name, postgres_settings):
        intake_query = sql.SQL(
            """
            SELECT
                COUNT(*) AS production_claims_received,
                ROUND(AVG(claim_amount)::numeric, 2) AS average_claim_amount,
                MIN(claim_date) AS min_claim_date,
                MAX(claim_date) AS max_claim_date
            FROM {view_name}
            """
        ).format(view_name=production_intake_view_identifier(postgres_settings))
        report["production_intake_overview"] = fetch_rows_as_dicts(intake_query, [], postgres_settings)[0]
    else:
        report["production_intake_overview"] = {
            "production_claims_received": 0,
            "note": "No production intake claims stored in PostgreSQL yet.",
        }

    return report


def write_sql_analyst_report(settings: PostgresSettings | None = None) -> dict[str, Any]:
    report = build_sql_analyst_report(settings=settings)
    SQL_ANALYST_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(write_report: bool = False) -> dict[str, Any]:
    settings = PostgresSettings.from_env()
    if write_report:
        report = write_sql_analyst_report(settings=settings)
        print(json.dumps(report, indent=2, default=str))
        return report
    claims_df = fetch_historical_claims(limit=5, settings=settings)
    preview = claims_df.head(5).to_dict(orient="records")
    print(json.dumps(preview, indent=2, default=str))
    return {"preview_rows": preview}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query historical claims or build a SQL analyst report.")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    main(write_report=args.write_report)
