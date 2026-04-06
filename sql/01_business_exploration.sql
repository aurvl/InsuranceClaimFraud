/*
Insurance Claim Counterfactual Simulator
10 business questions for SQL exploration, from basic profiling to high-value
fraud operations analysis.

Prerequisite:
  Run sql/02_create_analytics_views.sql first so the views exist.

Recommended execution:
  psql -d insurance_claims -f sql/01_business_exploration.sql

Each block is intentionally standalone so an analyst can run one question at a
time during EDA, dashboard prototyping, or an ad hoc investigation.
*/

-- ============================================================================
-- Q1. What is the size and overall fraud rate of the historical portfolio?
-- Why it matters:
--   This is the first sanity check before any deeper analysis. It tells us if
--   the portfolio is large enough, how imbalanced the fraud label is, and what
--   baseline analysts should expect.
-- ============================================================================
SELECT
	COUNT(*) as historical_claim_count,
    ROUND(AVG(
	 	CASE WHEN is_fraud then 1.0 ELSE 0.0 END
	 )::numeric, 4) as fraud_rate,
    MIN(claim_date) as first_claim_date,
    MAX(claim_date) as last_claim_date,
    ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount
FROM vw_historical_claims_enriched;

-- ============================================================================
-- Q2. Which claim types drive the most fraud volume and the highest fraud rate?
-- Why it matters:
--   Some claim categories may be frequent but low-risk, while others may be
--   rarer yet much riskier. This helps prioritize investigation playbooks.
-- ============================================================================
SELECT
	claim_type,
   COUNT(*) as claim_count,
   SUM(
		CASE WHEN is_fraud then 1 ELSE 0 END
	) as fraud_count,
   ROUND(AVG(
		CASE WHEN is_fraud then 1.0 ELSE 0.0 END)::numeric, 4
	) as fraud_rate,
   ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount
FROM vw_historical_claims_enriched
GROUP BY claim_type
ORDER BY fraud_rate DESC, fraud_count DESC, claim_count DESC;

-- ============================================================================
-- Q3. Which cities deserve more fraud monitoring attention?
-- Why it matters:
--   A fraud team often allocates audit and field review resources geographically.
--   This surfaces locations with both meaningful volume and elevated risk.
-- ============================================================================
SELECT
	location,
	COUNT(*) as claim_count,
	SUM(
		CASE WHEN is_fraud then 1 ELSE 0 END
	) as fraud_count,
	ROUND(AVG(
		CASE WHEN is_fraud then 1.0 ELSE 0.0 END
	)::numeric, 4) as fraud_rate,
	ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount
FROM vw_historical_claims_enriched
GROUP BY location
HAVING COUNT(*) >= 250 -- with at least 250 claims
ORDER BY fraud_rate DESC, claim_count DESC;

-- ============================================================================
-- Q4. Does fraud concentrate among high-amount claims?
-- Why it matters:
--   This quantifies whether very expensive claims deserve stronger routing to
--   senior investigators or additional document controls.
-- ============================================================================
WITH amount_buckets as (
	SELECT
		CASE
			WHEN claim_amount < 1000 then '< 1k'
			WHEN claim_amount < 3000 then '1k - 3k'
			WHEN claim_amount < 7000 then '3k - 7k'
			WHEN claim_amount < 12000 then '7k - 12k'
			ELSE '> 12k'
		END as claim_amount_bucket,
		is_fraud,
		claim_amount
	FROM vw_historical_claims_enriched
)
SELECT
	claim_amount_bucket,
	COUNT(*) as claim_count,
	ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount,
	ROUND(AVG(
		CASE WHEN is_fraud then 1.0 ELSE 0.0 END
	)::numeric, 4) as fraud_rate
FROM amount_buckets
GROUP BY claim_amount_bucket
ORDER BY
	CASE claim_amount_bucket
		WHEN '< 1k' then 1
		WHEN '1k - 3k' then 2
		WHEN '3k - 7k' then 3
		WHEN '7k - 12k' then 4
		ELSE 5
	END;

-- ============================================================================
-- Q5. How risky are claims filed very soon after policy start?
-- Why it matters:
--   Early-policy claims are a classic business red flag. This query measures how
--   much extra risk is concentrated in those early windows.
-- ============================================================================
WITH policy_age_buckets as (
	SELECT
		CASE
			WHEN policy_age_days <= 30 then '0 - 30d'
			WHEN policy_age_days <= 90 then '31 - 90d'
			WHEN policy_age_days <= 180 then '91 - 180d'
			WHEN policy_age_days <= 365 then '181 - 365d'
			ELSE '> 365d'
		END as policy_age_bucket,
		is_fraud,
		claim_amount
	FROM vw_historical_claims_enriched
)
SELECT
	policy_age_bucket,
	COUNT(*) as claim_count,
	ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount,
	ROUND(AVG(
		CASE WHEN is_fraud then 1.0 ELSE 0.0 END
	)::numeric, 4) as fraud_rate
FROM policy_age_buckets
GROUP BY policy_age_bucket
ORDER BY
	CASE policy_age_bucket
		WHEN '0 - 30d' then 1
		WHEN '31 - 90d' then 2
		WHEN '91 - 180d' then 3
		WHEN '181 - 365d' then 4
		ELSE 5
	END;

-- ============================================================================
-- Q6. How much does claim recency influence fraud risk?
-- Why it matters:
--   Repeat claims filed shortly after a previous one often signal abuse,
--   opportunism, or process anomalies that deserve review.
-- ============================================================================
WITH recency_buckets as (
	SELECT
		CASE
			WHEN time_since_last_claim_days IS NULL then 'first_claim'
			WHEN time_since_last_claim_days <= 30 then '0 - 30d'
			WHEN time_since_last_claim_days <= 90 then '31 - 90d'
			WHEN time_since_last_claim_days <= 180 then '91 - 180d'
			ELSE '> 180d'
		END as recency_bucket,
		is_fraud,
		num_previous_claims
	FROM vw_historical_claims_enriched
)
SELECT
	recency_bucket,
	COUNT(*) as claim_count,
	ROUND(AVG(num_previous_claims)::numeric, 2) as avg_previous_claims,
	ROUND(AVG(
		CASE WHEN is_fraud then 1.0 ELSE 0.0 END
	)::numeric, 4) as fraud_rate
FROM recency_buckets
GROUP BY recency_bucket
ORDER BY
	CASE recency_bucket
		WHEN 'first_claim' THEN 1
		WHEN '0 - 30d' THEN 2
		WHEN '31 - 90d' THEN 3
		WHEN '91 - 180d' THEN 4
		ELSE 5
	END;

-- ============================================================================
-- Q7. Which service providers are structurally suspicious?
-- Why it matters:
--   Provider risk is operationally powerful because it supports partner audits,
--   claim routing, and enhanced verification rules.
-- ============================================================================
WITH provider_stats as (
	SELECT
		service_provider_id,
		COUNT(*) as claim_count,
		SUM(
			CASE WHEN is_fraud then 1 ELSE 0 END
		) as fraud_count,
		ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount,
		ROUND(AVG(
			CASE WHEN is_fraud then 1.0 ELSE 0.0 END
		)::numeric, 4) as fraud_rate
	FROM vw_historical_claims_enriched
	GROUP BY service_provider_id
)
SELECT
	service_provider_id,
	claim_count,
	fraud_count,
	fraud_rate,
	avg_claim_amount
FROM provider_stats
WHERE claim_count >= 25
ORDER BY fraud_rate DESC, claim_count DESC
LIMIT 15;

-- ============================================================================
-- Q8. Which combinations of claim type, weather, and provider look abnormal?
-- Why it matters:
--   Business users rarely investigate one feature alone. This query searches for
--   suspicious interaction effects that can inspire new rules or QA checks.
-- ============================================================================
WITH combination_stats as (
	SELECT
		claim_type,
		weather_condition,
		service_provider_id,
		COUNT(*) as claim_count,
		SUM(
			CASE WHEN is_fraud then 1 ELSE 0 END
		) as fraud_count,
		ROUND(AVG(
			CASE WHEN is_fraud then 1.0 ELSE 0.0 END
		)::numeric, 4) as fraud_rate,
		ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount
	FROM vw_historical_claims_enriched
	GROUP BY claim_type, weather_condition, service_provider_id
)
SELECT
	claim_type,
	weather_condition,
	service_provider_id,
	claim_count,
	fraud_count,
	fraud_rate,
	avg_claim_amount
FROM combination_stats
WHERE claim_count >= 10
ORDER BY fraud_rate DESC, claim_count DESC
LIMIT 20;

-- ============================================================================
-- Q9. How is fraud evolving over time at a monthly level?
-- Why it matters:
--   Trend analysis helps teams distinguish a stable process from regime changes
--   that may require updated rules, retraining, or operational escalation.
-- ============================================================================
WITH monthly_metrics as (
	SELECT
		DATE_TRUNC('month', claim_date)::date as claim_month,
		COUNT(*) AS claim_count,
		SUM(
			CASE WHEN is_fraud then 1 ELSE 0 END
		) as fraud_count,
		ROUND(AVG(
			CASE WHEN is_fraud then 1.0 ELSE 0.0 END
		)::numeric, 4) as fraud_rate,
		ROUND(AVG(claim_amount)::numeric, 2) as avg_claim_amount
	FROM vw_historical_claims_enriched
	GROUP BY DATE_TRUNC('month', claim_date)
)
SELECT
	claim_month,
	claim_count,
	fraud_count,
	fraud_rate,
	avg_claim_amount,
	LAG(fraud_rate) OVER (ORDER BY claim_month) as previous_month_fraud_rate,
	ROUND(
		(fraud_rate - LAG(fraud_rate) OVER (ORDER BY claim_month))::numeric, 4
	) as fraud_rate_delta
FROM monthly_metrics
ORDER BY claim_month;

-- ============================================================================
-- Q10. Which historical claims look most similar to the kind of cases that
-- could justify a counterfactual intervention?
-- Why it matters:
--   This is the closest SQL bridge between fraud detection and decision support.
--   It finds high-risk historical patterns where a small change in amount,
--   provider, or recency might matter operationally.
-- ============================================================================
WITH flagged_claims as (
	SELECT
		claim_id,
		customer_id,
		claim_date,
		claim_type,
		service_provider_id,
		claim_amount,
		policy_age_days,
		time_since_last_claim_days,
		early_policy_claim_flag,
		high_amount_flag,
		recent_claim_flag,
		suspicious_provider_flag,
		is_fraud,
		(
			CASE WHEN high_amount_flag then 1 ELSE 0 END +
			CASE WHEN recent_claim_flag then 1 ELSE 0 END +
			CASE WHEN suspicious_provider_flag then 1 ELSE 0 END +
			CASE WHEN early_policy_claim_flag then 1 ELSE 0 END
		) as suspicious_signal_count
	FROM vw_historical_claims_enriched
)
SELECT
	claim_id,
	customer_id,
	claim_date,
	claim_type,
	service_provider_id,
	claim_amount,
	policy_age_days,
	time_since_last_claim_days,
	suspicious_signal_count,
	is_fraud
FROM flagged_claims
WHERE suspicious_signal_count >= 3
ORDER BY suspicious_signal_count DESC, claim_amount DESC, claim_date DESC
LIMIT 25;
