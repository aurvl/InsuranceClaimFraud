/*
Insurance Claim Counterfactual Simulator
Rebuild the canonical analytics views consumed by Python and BI users.

Run with psql for the default portfolio schema:
  psql -d insurance_claims -f sql/02_create_analytics_views.sql

This script intentionally centralizes the SQL shaping logic in PostgreSQL so
the Python layer mostly issues simple SELECT statements against stable views.
*/

BEGIN;

DROP VIEW IF EXISTS vw_historical_claims_enriched CASCADE;
DROP VIEW IF EXISTS vw_production_claim_intake_enriched CASCADE;
DROP VIEW IF EXISTS vw_production_claim_decisions_enriched CASCADE;

CREATE VIEW vw_historical_claims_enriched as
WITH claim_base as (
	SELECT
		cl.claim_id,
		cl.customer_id,
		cl.policy_id,
		cl.claim_date,
		cl.claim_amount,
		cl.claim_type,
		cl.num_previous_claims,
		cl.time_since_last_claim_days,
		cl.service_provider_id,
		cl.weather_condition,
		cl.is_fraud,
		cu.customer_age,
		cu.location,
		p.policy_start_date
	FROM claims as cl
	INNER JOIN customers as cu
		ON cu.customer_id = cl.customer_id
	INNER JOIN policies as p
		ON p.policy_id = cl.policy_id
),
derived_metrics as (
	SELECT
		claim_id,
		customer_id,
		policy_id,
		claim_date,
		policy_start_date,
		(claim_date - policy_start_date) as policy_age_days,
		claim_amount,
		claim_type,
		customer_age,
		location,
		num_previous_claims,
		time_since_last_claim_days,
		service_provider_id,
		weather_condition,
		is_fraud,
		CASE
			WHEN policy_start_date IS NULL or claim_date IS NULL then NULL
			WHEN (claim_date - policy_start_date) <= 30 then TRUE
			ELSE FALSE
		END as early_policy_claim_flag,
		CASE
			WHEN claim_amount >= 12000 then TRUE
			ELSE FALSE
		END as high_amount_flag,
		CASE
			WHEN time_since_last_claim_days IS NOT NULL and time_since_last_claim_days <= 45 then TRUE
			ELSE FALSE
		END as recent_claim_flag,
		CASE
			WHEN service_provider_id IN (
				'SP-003', 
				'SP-017', 
				'SP-044', 
				'SP-058', 
				'SP-071', 
				'SP-089', 
				'SP-111'
			) then TRUE
			ELSE FALSE
		END as suspicious_provider_flag
		FROM claim_base
)
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
	is_fraud,
	early_policy_claim_flag,
	high_amount_flag,
	recent_claim_flag,
	suspicious_provider_flag,
	ROUND(claim_amount / NULLIF(policy_age_days, 0), 4) as claim_amount_to_policy_age
FROM derived_metrics;

CREATE VIEW vw_production_claim_intake_enriched as
WITH intake_base as (
	SELECT
		pi.claim_id,
		pi.customer_id,
		pi.policy_id,
		pi.claim_date,
		pi.policy_start_date,
		pi.policy_age_days,
		pi.claim_amount,
		pi.claim_type,
		pi.customer_age,
		pi.location,
		pi.num_previous_claims,
		pi.time_since_last_claim_days,
		pi.service_provider_id,
		pi.weather_condition,
		pi.document_extraction_mode,
		pi.chosen_sources,
		pi.mismatches,
		pi.package_metadata,
		pi.structured_claim,
		pi.created_at,
		pi.updated_at
	FROM production_claim_intake as pi
)
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
	document_extraction_mode,
	chosen_sources,
	mismatches,
	package_metadata,
	structured_claim,
	CASE
		WHEN claim_amount >= 12000 then TRUE
		ELSE FALSE
	END as high_amount_flag,
	CASE
		WHEN time_since_last_claim_days IS NOT NULL and time_since_last_claim_days <= 45 then TRUE
		ELSE FALSE
	END as recent_claim_flag,
	CASE
		WHEN service_provider_id IN (
			'SP-003', 
			'SP-017', 
			'SP-044', 
			'SP-058', 
			'SP-071', 
			'SP-089', 
			'SP-111'
		) then TRUE
		ELSE FALSE
	END as suspicious_provider_flag,
	created_at,
	updated_at
FROM intake_base;

CREATE VIEW vw_production_claim_decisions_enriched as
WITH decision_base as (
	SELECT
		pd.claim_id,
		pd.customer_id,
		pd.policy_id,
		pd.claim_date,
		pd.policy_start_date,
		pd.policy_age_days,
		pd.claim_amount,
		pd.claim_type,
		pd.customer_age,
		pd.location,
		pd.num_previous_claims,
		pd.time_since_last_claim_days,
		pd.service_provider_id,
		pd.weather_condition,
		pd.fraud_probability,
		pd.model_decision,
		pd.decision_threshold,
		pd.intake_mode,
		pd.package_directory,
		pd.record_path,
		pd.chosen_sources,
		pd.mismatch_fields,
		pd.package_metadata,
		pd.prediction_metadata,
		pd.created_at,
		pd.updated_at
	FROM production_claim_decisions as pd
)
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
	fraud_probability,
	model_decision,
	decision_threshold,
	intake_mode,
	package_directory,
	record_path,
	chosen_sources,
	mismatch_fields,
	package_metadata,
	prediction_metadata,
	CASE
		WHEN fraud_probability >= decision_threshold then TRUE
		ELSE FALSE
	END as above_threshold_flag,
	created_at,
	updated_at
FROM decision_base;

COMMIT;
