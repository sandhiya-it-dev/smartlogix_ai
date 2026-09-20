from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATABASE_URL, PROCESSED_DIR


def get_engine(url: str = DATABASE_URL):
    return create_engine(url, future=True)


def persist_tables(tables: dict[str, pd.DataFrame], url: str = DATABASE_URL) -> None:
    prepared = {}
    for name, df in tables.items():
        safe = df.copy()
        for col in safe.columns:
            if safe[col].map(lambda x: isinstance(x, (list, dict))).any():
                safe[col] = safe[col].map(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        safe.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        prepared[name] = safe

    if url.startswith("sqlite:///"):
        target = Path(url.removeprefix("sqlite:///"))
        temporary = target.with_suffix(".building.db")
        temporary.unlink(missing_ok=True)
        conn = sqlite3.connect(temporary)
        try:
            for name, safe in prepared.items():
                safe.to_sql(name, conn, if_exists="replace", index=False, chunksize=2000)
            conn.execute("CREATE INDEX ix_orders_order_id ON orders(order_id)")
            conn.execute("CREATE INDEX ix_logs_order_id ON delivery_logs(order_id)")
            conn.execute("CREATE INDEX ix_reviews_product_id ON reviews(product_id)")
            conn.commit()
            check = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise RuntimeError(f"SQLite integrity check failed: {check}")
        finally:
            conn.close()
        os.replace(temporary, target)
    else:
        engine = get_engine(url)
        for name, safe in prepared.items():
            safe.to_sql(name, engine, if_exists="replace", index=False, chunksize=2000)
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_orders_order_id ON orders(order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_logs_order_id ON delivery_logs(order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_reviews_product_id ON reviews(product_id)"))
        engine.dispose()


def query_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


if __name__ == "__main__":
    from .cleaning import clean_all
    from .config import RAW_DIR

    print("Creating the SmartLogix database from cleaned datasets...")
    cleaned_tables = clean_all(RAW_DIR)
    persist_tables(cleaned_tables)

    table_summary = query_df(
        "SELECT name AS table_name FROM sqlite_master "
        "WHERE type='table' ORDER BY name"
    )
    print("\nDatabase created successfully.")
    print(f"Database URL: {DATABASE_URL}")
    print(f"Tables created ({len(table_summary)}):")
    for table_name in table_summary["table_name"]:
        row_count = int(query_df(f'SELECT COUNT(*) AS total FROM "{table_name}"').iloc[0]["total"])
        print(f"  - {table_name}: {row_count} rows")
