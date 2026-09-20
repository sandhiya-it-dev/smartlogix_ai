from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def load_json_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except json.JSONDecodeError:
        with path.open(encoding="utf-8") as stream:
            value = [json.loads(line) for line in stream if line.strip()]
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        lists = [item for item in value.values() if isinstance(item, list)]
        records = max(lists, key=len) if lists else [value]
    else:
        raise ValueError(f"Unsupported JSON structure: {path}")
    return pd.json_normalize(records)


def parse_number(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else np.nan


def parse_weight_kg(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(",", "")
    number = parse_number(text)
    if np.isnan(number):
        return np.nan
    if "mg" in text:
        return number / 1_000_000
    if "kg" in text:
        return number
    if " g" in text or text.endswith("g"):
        return number / 1000
    return number


def parse_bool(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return 1.0
    if text in FALSE_VALUES:
        return 0.0
    return np.nan


def parse_datetime(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    unix_mask = numeric.between(1_000_000_000, 2_000_000_000)
    result = pd.to_datetime(raw, errors="coerce", dayfirst=True, utc=True, format="mixed")
    if unix_mask.any():
        result.loc[unix_mask] = pd.to_datetime(numeric.loc[unix_mask], unit="s", utc=True)
    return result.dt.tz_localize(None)


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()


def clean_identifier_duplicates(df: pd.DataFrame, key: str) -> pd.DataFrame:
    # Primary-key deduplication also works when other columns contain lists/dicts.
    return df.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)


if __name__ == "__main__":
    print("SmartLogix data utility test")
    print(f"Number parsing : INR 1,250.50 -> {parse_number('INR 1,250.50')}")
    print(f"Weight parsing : 750 g -> {parse_weight_kg('750 g')} kg")
    print(f"Boolean parsing: Yes -> {int(parse_bool('Yes'))}")

    sample_dates = pd.Series(["16-09-2026", "2026-09-17"])
    print("Date parsing   :", parse_datetime(sample_dates).dt.strftime("%Y-%m-%d").tolist())

    sample_ids = pd.DataFrame({"id": ["A", "A", "B"], "value": [1, 2, 3]})
    cleaned = clean_identifier_duplicates(sample_ids, "id")
    print(f"Duplicate test : {len(sample_ids)} rows -> {len(cleaned)} rows")
