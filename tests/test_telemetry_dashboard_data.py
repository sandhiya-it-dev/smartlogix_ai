import pandas as pd


def test_dashboard_numeric_conversion_handles_text_values():
    health = pd.DataFrame({
        "battery_health_pct": ["91.6", "bad"],
        "motor_temp_c": ["83.7", "72.1"],
        "vibration_rms": ["2.991", None],
        "maintenance_required": ["0", "1"],
    })
    for column in health.columns:
        health[column] = pd.to_numeric(health[column], errors="coerce")
    health = health.dropna(subset=["battery_health_pct", "motor_temp_c", "vibration_rms"])
    assert len(health) == 1
    assert health.iloc[0]["vibration_rms"] == 2.991
