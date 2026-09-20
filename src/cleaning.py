from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_utils import (clean_identifier_duplicates, load_json_records, normalize_text,
                         parse_bool, parse_datetime, parse_number, parse_weight_kg)


def clean_orders(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path, low_memory=False), "order_id")
    df["order_date"] = parse_datetime(df["order_date"])
    for col in ["distance_km", "order_value_inr", "delivery_cost_inr", "promised_eta_hours", "actual_delivery_hours"]:
        df[col] = df[col].map(parse_number)
    df["package_weight_kg"] = df["package_weight"].map(parse_weight_kg)
    for col in ["is_fragile", "is_hazmat", "cold_chain_required"]:
        df[col] = df[col].map(parse_bool).fillna(0).astype(int)
    for col in ["origin_city", "destination_city", "weather_condition_at_dest", "delivery_priority", "payment_mode", "order_status"]:
        df[col] = normalize_text(df[col])
    mode = normalize_text(df["transport_mode"]).str.replace("_", " ", regex=False)
    valid_modes = {"Drone", "Bike", "Van", "Truck", "Air Cargo", "Ship"}
    df["transport_mode"] = mode.where(mode.isin(valid_modes), pd.NA)
    df["on_time"] = (df["actual_delivery_hours"] <= df["promised_eta_hours"]).astype("Int64")
    df["delay_hours"] = (df["actual_delivery_hours"] - df["promised_eta_hours"]).clip(lower=0)
    df["cost_per_km"] = df["delivery_cost_inr"] / df["distance_km"].replace(0, np.nan)
    return df.drop(columns=["package_weight"])


def clean_customers(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path), "customer_id")
    df["signup_date"] = parse_datetime(df["signup_date"])
    df["is_prime_member"] = df["is_prime_member"].map(parse_bool).fillna(0).astype(int)
    df["customer_segment"] = normalize_text(df["customer_segment"])
    for c in ["lifetime_orders", "avg_rating_given"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean_fleet(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path), "vehicle_id")
    df["vehicle_type"] = normalize_text(df["vehicle_type"])
    status = normalize_text(df["fleet_status"]).replace({"In Service":"Active", "Under Maintenance":"Maintenance"})
    df["fleet_status"] = status
    for c in ["purchase_date", "last_service_date", "insurance_expiry"]: df[c] = parse_datetime(df[c])
    return df


def clean_telemetry(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path), "flight_id")
    df["flight_timestamp"] = parse_datetime(df["flight_timestamp"])
    numeric_columns = [
        "flight_duration_min", "cumulative_flight_hours", "battery_cycles",
        "battery_start_pct", "battery_end_pct", "battery_health_pct",
        "motor_temp_c", "vibration_rms", "payload_kg", "max_altitude_m",
        "wind_speed_kmph", "rotor_rpm_avg", "route_deviation_m",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gps_signal_quality"] = normalize_text(df["gps_signal_quality"])
    df["maintenance_required"] = pd.to_numeric(df["maintenance_required"], errors="coerce").fillna(0).astype(int)
    return df


def clean_maintenance(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path), "work_order_id")
    df["service_date"] = parse_datetime(df["service_date"])
    df["cost_inr"] = df["cost_inr"].map(parse_number)
    df["failure_reported"] = df["failure_reported"].map(parse_bool)
    return df


def clean_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates().rename(columns={"Date":"date", "City Name":"city", "Temp Unit":"temp_unit", "humidity_%":"humidity_pct", "Precipitation (mm)":"precipitation_mm", "Visibility_KM":"visibility_km"})
    df["date"] = parse_datetime(df["date"])
    unit = df["temp_unit"].astype("string").str.upper()
    is_fahrenheit = unit.eq("F").fillna(False).to_numpy(dtype=bool)
    numeric_temp = pd.to_numeric(df["temp"], errors="coerce")
    df["temp_c"] = np.where(is_fahrenheit, (numeric_temp - 32) * 5/9, numeric_temp)
    df["condition"] = normalize_text(df["condition"])
    df["storm_alert"] = df["storm_alert"].map(parse_bool).fillna(0).astype(int)
    return df.drop(columns=["temp", "temp_unit"])


def clean_traffic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    df["record_date"] = parse_datetime(df["record_date"])
    df["hour_of_day"] = df["hour_of_day"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    for c in ["incident_reported", "road_closure"]: df[c] = df[c].map(parse_bool).fillna(0).astype(int)
    return df


def clean_logs(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(pd.read_csv(path, low_memory=False), "log_id")
    df["event_timestamp"] = parse_datetime(df["event_timestamp"])
    df["event_type"] = df["event_type"].astype("string").str.strip().str.upper()
    return df


def clean_products(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(load_json_records(path), "product_id")
    df = df.rename(columns={"specs.weight_kg":"weight_kg", "price.amount":"price_inr", "price.currency":"currency", "specs.battery_included":"battery_included"})
    df["price_inr"] = df["price_inr"].map(parse_number)
    df["currency"] = df["currency"].astype("string").str.upper()
    df["launch_date"] = parse_datetime(df["launch_date"])
    df["tags"] = df["tags"].map(lambda x: ", ".join(x) if isinstance(x, list) else "")
    return df.drop(columns=["specs"], errors="ignore")


def clean_reviews(path: Path) -> pd.DataFrame:
    df = clean_identifier_duplicates(load_json_records(path), "review_id")
    rating_words = {"one":1,"two":2,"three":3,"four":4,"five":5}
    df["rating"] = df["rating"].map(lambda x: rating_words.get(str(x).lower(), parse_number(x)))
    df["review_date"] = parse_datetime(df["review_date"])
    df["verified_purchase"] = df["verified_purchase"].map(parse_bool).fillna(0).astype(int)
    inferred = pd.cut(df["rating"], bins=[0,2,3,5], labels=["neg","neu","pos"], include_lowest=True).astype("string")
    df["sentiment_label"] = df["sentiment_label"].fillna(inferred)
    df["delivery_mode_experienced"] = normalize_text(df["delivery_mode_experienced"])
    return df


def clean_routes(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_json_records(path)
    waypoints = []
    for _, row in df.iterrows():
        for point in row.get("waypoints", []) or []:
            waypoints.append({"route_id":row["route_id"], **point})
    wp = pd.DataFrame(waypoints)
    if not wp.empty: wp["ts"] = parse_datetime(wp["ts"])
    df["order_ids"] = df["order_ids"].map(lambda x: ",".join(x) if isinstance(x,list) else "")
    df["waypoint_count"] = df["waypoints"].map(lambda x: len(x) if isinstance(x,list) else 0)
    df["fuel_or_energy_value"] = df["fuel_or_energy_used"].map(parse_number)
    return df.drop(columns=["waypoints"]), wp


def clean_all(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    p = Path(raw_dir)
    routes, waypoints = clean_routes(p / "gps_routes.json")
    return {
        "orders": clean_orders(p / "orders.csv"), "customers": clean_customers(p / "customers.csv"),
        "fleet": clean_fleet(p / "fleet_vehicles.csv"), "drone_telemetry": clean_telemetry(p / "drone_telemetry.csv"),
        "maintenance": clean_maintenance(p / "maintenance_history.csv"), "weather": clean_weather(p / "weather_data.csv"),
        "traffic": clean_traffic(p / "traffic_data.csv"), "delivery_logs": clean_logs(p / "delivery_logs.csv"),
        "products": clean_products(p / "product_catalog.json"), "reviews": clean_reviews(p / "customer_reviews.json"),
        "routes": routes, "waypoints": waypoints,
    }


if __name__ == "__main__":
    from .config import RAW_DIR

    print("Cleaning SmartLogix raw datasets...")
    cleaned_tables = clean_all(RAW_DIR)
    print("\nCleaning completed successfully.")
    print("-" * 46)
    print(f"{'Table':<25}{'Rows':>10}{'Columns':>10}")
    print("-" * 46)
    for table_name, table in cleaned_tables.items():
        print(f"{table_name:<25}{len(table):>10}{len(table.columns):>10}")
    print("-" * 46)
