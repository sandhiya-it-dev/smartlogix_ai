from __future__ import annotations

import numpy as np
import pandas as pd


def add_mode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create features required by the transport-mode classifier."""

    data = frame.copy()

    numeric_columns = [
        "distance_km",
        "package_weight_kg",
        "quantity",
        "is_fragile",
        "is_hazmat",
        "cold_chain_required",
        "order_value_inr",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["distance_log"] = np.log1p(
        data["distance_km"].clip(lower=0)
    )

    data["weight_log"] = np.log1p(
        data["package_weight_kg"].clip(lower=0)
    )

    data["value_log"] = np.log1p(
        data["order_value_inr"].clip(lower=0)
    )

    data["same_city"] = (
        data["origin_city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        data["destination_city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    ).astype(int)

    data["drone_eligible"] = (
        (data["distance_km"] <= 25)
        & (data["package_weight_kg"] <= 5)
        & (data["is_hazmat"] == 0)
        & (data["cold_chain_required"] == 0)
    ).astype(int)

    data["long_distance"] = (
        data["distance_km"] > 500
    ).astype(int)

    data["heavy_package"] = (
        data["package_weight_kg"] > 100
    ).astype(int)

    return data