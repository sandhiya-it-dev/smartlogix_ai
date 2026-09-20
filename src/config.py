from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROCESSED_DIR / 'smartlogix.db'}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("SmartLogix configuration loaded successfully.")
    print(f"Project root : {ROOT}")
    print(f"Raw data     : {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Models       : {MODEL_DIR}")
    print(f"Database     : {DATABASE_URL}")
