from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL


# ==========================================================
# PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        str(ROOT / "models"),
    )
)


# ==========================================================
# ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv(ROOT / ".env")


POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = int(
    os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
)

POSTGRES_DB = os.getenv(
    "POSTGRES_DB",
    "smartlogix",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "smartlogix_user",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD"
)

if not POSTGRES_PASSWORD:
    raise RuntimeError(
        "POSTGRES_PASSWORD is missing. "
        "Add it to the project .env file."
    )


# ==========================================================
# POSTGRESQL CONNECTION URL
# ==========================================================

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
)


# ==========================================================
# CREATE LOCAL PROJECT DIRECTORIES
# ==========================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# CONFIGURATION TEST
# ==========================================================

if __name__ == "__main__":
    print(
        "SmartLogix configuration loaded successfully."
    )

    print(f"Project root  : {ROOT}")
    print(f"Raw data      : {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Models        : {MODEL_DIR}")
    print(f"Database type : PostgreSQL")
    print(f"Database host : {POSTGRES_HOST}")
    print(f"Database port : {POSTGRES_PORT}")
    print(f"Database name : {POSTGRES_DB}")
    print(f"Database user : {POSTGRES_USER}")