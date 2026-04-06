from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load environment variables from a .env file at the project root (if present)
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GROUND_TRUTH_DIR = PROCESSED_DATA_DIR / "ground_truth"
TRAIN_STRUCTURED_DIR = RAW_DATA_DIR / "train" / "csv"
PROD_STRUCTURED_DIR = RAW_DATA_DIR / "prod" / "csv"
GROUND_TRUTH_TRAIN_PATH = GROUND_TRUTH_DIR / "claims_train_ground_truth.json"
GROUND_TRUTH_PROD_PATH = GROUND_TRUTH_DIR / "new_claims_ground_truth.json"
TRAIN_PDF_DIR = RAW_DATA_DIR / "train" / "pdf"
TRAIN_IMAGE_DIR = RAW_DATA_DIR / "train" / "images"
TRAIN_CLIENT_DIR = RAW_DATA_DIR / "train" / "clients"
PROD_PDF_DIR = RAW_DATA_DIR / "prod" / "pdf"
PROD_IMAGE_DIR = RAW_DATA_DIR / "prod" / "images"
PROD_CLIENT_DIR = RAW_DATA_DIR / "prod" / "clients"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "deployment_champion.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "deployment_champion.json"
EVALUATION_PATH = PROCESSED_DATA_DIR / "evaluation_metrics.json"
HISTORICAL_TEST_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "historical_test_predictions.csv"
FEATURE_IMPORTANCE_PATH = PROCESSED_DATA_DIR / "feature_importances.csv"
DATA_SUMMARY_PATH = PROCESSED_DATA_DIR / "data_generation_summary.json"
TRAIN_EXTRACTION_AUDIT_JSON_PATH = PROCESSED_DATA_DIR / "train_package_extraction_audit.json"
PROD_EXTRACTION_AUDIT_JSON_PATH = PROCESSED_DATA_DIR / "prod_package_extraction_audit.json"
SQL_ANALYST_REPORT_PATH = PROCESSED_DATA_DIR / "sql_analyst_report.json"
REPORT_PATH = PROJECT_ROOT / "report" / "insurance_counterfactual_study_report.docx"
EDA_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_exploration_and_eda.ipynb"
MODELING_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_modeling_and_counterfactuals.ipynb"
SQL_DIR = PROJECT_ROOT / "sql"
SQL_EXPLORATION_SCRIPT_PATH = SQL_DIR / "01_business_exploration.sql"
SQL_VIEWS_SCRIPT_PATH = SQL_DIR / "02_create_analytics_views.sql"
PRODUCTION_PACKAGE_DIR = RAW_DATA_DIR / "prod" / "incoming_packages"
PRODUCTION_PACKAGE_RECORD_DIR = PROCESSED_DATA_DIR / "production_package_records"
DEVELOPMENT_CUTOFF_DATE = date(2024, 10, 2)
PRODUCTION_START_DATE = date(2024, 10, 3)
DEFAULT_CLAIMS_TABLE = os.getenv("POSTGRES_TABLE", "claims")
DEFAULT_CUSTOMERS_TABLE = os.getenv("POSTGRES_CUSTOMERS_TABLE", "customers")
DEFAULT_POLICIES_TABLE = os.getenv("POSTGRES_POLICIES_TABLE", "policies")
DEFAULT_PRODUCTION_INTAKE_TABLE = os.getenv(
    "POSTGRES_PRODUCTION_INTAKE_TABLE",
    "production_claim_intake",
)
DEFAULT_PRODUCTION_DECISIONS_TABLE = os.getenv(
    "POSTGRES_PRODUCTION_DECISIONS_TABLE",
    "production_claim_decisions",
)
DEFAULT_HISTORICAL_VIEW = os.getenv("POSTGRES_HISTORICAL_VIEW", "vw_historical_claims_enriched")
DEFAULT_PRODUCTION_INTAKE_VIEW = os.getenv("POSTGRES_PRODUCTION_INTAKE_VIEW", "vw_production_claim_intake_enriched")
DEFAULT_PRODUCTION_DECISIONS_VIEW = os.getenv(
    "POSTGRES_PRODUCTION_DECISIONS_VIEW",
    "vw_production_claim_decisions_enriched",
)
DEFAULT_DECISION_THRESHOLD = float(os.getenv("MODEL_DECISION_THRESHOLD", "0.5"))

RECENT_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "recent_production_predictions.csv"

CLAIM_TYPES = [
    "collision",
    "theft",
    "windshield",
    "water_damage",
    "fire",
    "bodily_injury",
    "hail_damage",
]

LOCATIONS = [
    "Paris",
    "Marseille",
    "Lyon",
    "Toulouse",
    "Nice",
    "Nantes",
    "Strasbourg",
    "Montpellier",
    "Bordeaux",
    "Lille",
    "Rennes",
    "Reims",
    "Le Havre",
    "Saint-Étienne",
    "Toulon",
    "Grenoble",
    "Dijon",
    "Angers",
    "Nîmes",
    "Villeurbanne",
    "Clermont-Ferrand",
    "Le Mans",
    "Aix-en-Provence",
    "Brest",
    "Tours",
    "Amiens",
    "Limoges",
    "Perpignan",
    "Metz",
    "Orléans",
    "Cannes",
    "Annecy",
    "La Rochelle",
    "Saint-Malo",
    "Caen",
    "Ajaccio",
    "Nancy",
]

# Realistic sampling weights for the locations (favours large / tourist cities).
# These are normalized to sum to 1.0 and must match length of `LOCATIONS`.
_raw_location_weights = np.array([
    # Major hub
    12.0,  # Paris
    # Large cities
    3.8,  # Marseille
    4.2,  # Lyon
    3.4,  # Toulouse
    3.0,  # Nice
    2.0,  # Nantes
    1.9,  # Strasbourg
    1.8,  # Montpellier
    2.2,  # Bordeaux
    2.2,  # Lille
    # Medium cities
    1.6,  # Rennes
    1.2,  # Reims
    1.0,  # Le Havre
    1.0,  # Saint-Étienne
    1.0,  # Toulon
    1.0,  # Grenoble
    0.9,  # Dijon
    0.9,  # Angers
    0.8,  # Nîmes
    0.7,  # Villeurbanne
    0.7,  # Clermont-Ferrand
    0.7,  # Le Mans
    1.0,  # Aix-en-Provence
    0.6,  # Brest
    0.7,  # Tours
    0.6,  # Amiens
    0.5,  # Limoges
    0.6,  # Perpignan
    0.6,  # Metz
    0.6,  # Orléans
    # Additional / tourist / regional
    1.3,  # Cannes
    0.7,  # Annecy
    0.6,  # La Rochelle
    0.5,  # Saint-Malo
    0.7,  # Caen
    0.5,  # Ajaccio
    0.7,  # Nancy
])
if len(_raw_location_weights) != len(LOCATIONS):
    LOCATION_WEIGHTS = np.full(len(LOCATIONS), 1.0 / len(LOCATIONS))
else:
    LOCATION_WEIGHTS = (_raw_location_weights / float(_raw_location_weights.sum()))

WEATHER_CONDITIONS = [
    "clear",
    "rain",
    "storm",
    "snow",
    "hail",
    "fog",
    "heatwave",
]

SUSPICIOUS_PROVIDERS = [ # with more than 10% fraud rate
    "SP-003",
    "SP-017",
    "SP-044",
    "SP-058",
    "SP-071",
    "SP-089",
    "SP-111",
    "SP-115",
]


@dataclass(frozen=True)
class PostgresSettings:
    host: str
    port: int
    database: str
    user: str
    password: str
    table_name: str = DEFAULT_CLAIMS_TABLE
    customers_table_name: str = DEFAULT_CUSTOMERS_TABLE
    policies_table_name: str = DEFAULT_POLICIES_TABLE
    production_intake_table_name: str = DEFAULT_PRODUCTION_INTAKE_TABLE
    production_table_name: str = DEFAULT_PRODUCTION_DECISIONS_TABLE
    historical_view_name: str = DEFAULT_HISTORICAL_VIEW
    production_intake_view_name: str = DEFAULT_PRODUCTION_INTAKE_VIEW
    production_view_name: str = DEFAULT_PRODUCTION_DECISIONS_VIEW
    
    @classmethod
    def from_env(cls) -> "PostgresSettings":
        def clean_env(name: str, default: str) -> str:
            raw_value = os.getenv(name, default)
            return raw_value.strip().strip("\"'")

        return cls(
            host=clean_env("POSTGRES_HOST", "localhost"),
            port=int(clean_env("POSTGRES_PORT", "5432")),
            database=clean_env("POSTGRES_DB", "insurance_claims"),
            user=clean_env("POSTGRES_USER", "postgres"),
            password=clean_env("POSTGRES_PASSWORD", "postgres"),
            table_name=clean_env("POSTGRES_TABLE", DEFAULT_CLAIMS_TABLE),
            customers_table_name=clean_env("POSTGRES_CUSTOMERS_TABLE", DEFAULT_CUSTOMERS_TABLE),
            policies_table_name=clean_env("POSTGRES_POLICIES_TABLE", DEFAULT_POLICIES_TABLE),
            production_intake_table_name=clean_env(
                "POSTGRES_PRODUCTION_INTAKE_TABLE",
                DEFAULT_PRODUCTION_INTAKE_TABLE,
            ),
            production_table_name=clean_env(
                "POSTGRES_PRODUCTION_DECISIONS_TABLE",
                DEFAULT_PRODUCTION_DECISIONS_TABLE,
            ),
            historical_view_name=clean_env("POSTGRES_HISTORICAL_VIEW", DEFAULT_HISTORICAL_VIEW),
            production_intake_view_name=clean_env(
                "POSTGRES_PRODUCTION_INTAKE_VIEW",
                DEFAULT_PRODUCTION_INTAKE_VIEW,
            ),
            production_view_name=clean_env(
                "POSTGRES_PRODUCTION_DECISIONS_VIEW",
                DEFAULT_PRODUCTION_DECISIONS_VIEW,
            ),
        )

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


def ensure_project_directories() -> None:
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        GROUND_TRUTH_DIR,
        TRAIN_STRUCTURED_DIR,
        PROD_STRUCTURED_DIR,
        TRAIN_PDF_DIR,
        TRAIN_IMAGE_DIR,
        TRAIN_CLIENT_DIR,
        PROD_PDF_DIR,
        PROD_IMAGE_DIR,
        PROD_CLIENT_DIR,
        PRODUCTION_PACKAGE_DIR,
        PRODUCTION_PACKAGE_RECORD_DIR,
        MODEL_DIR,
        SQL_DIR,
        REPORT_PATH.parent,
        EDA_NOTEBOOK_PATH.parent,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
