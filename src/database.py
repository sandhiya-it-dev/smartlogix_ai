from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import DATABASE_URL, PROCESSED_DIR


# ==========================================================
# POSTGRESQL ENGINE
# ==========================================================

ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)


def get_engine() -> Engine:
    """
    Return the shared PostgreSQL SQLAlchemy engine.
    """

    return ENGINE


# ==========================================================
# DATAFRAME PREPARATION
# ==========================================================

def _prepare_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert lists and dictionaries into JSON strings so they
    can be stored safely in PostgreSQL text columns.
    """

    safe = dataframe.copy()

    for column in safe.columns:
        contains_nested_values = safe[column].map(
            lambda value: isinstance(
                value,
                (list, dict),
            )
        ).any()

        if contains_nested_values:
            safe[column] = safe[column].map(
                lambda value: json.dumps(
                    value,
                    default=str,
                )
                if isinstance(value, (list, dict))
                else value
            )

    return safe


# ==========================================================
# SAVE TABLES
# ==========================================================

def persist_tables(
    tables: dict[str, pd.DataFrame],
) -> None:
    """
    Save cleaned DataFrames as CSV files and PostgreSQL tables.

    Existing PostgreSQL tables with the same names are replaced.
    """

    prepared_tables: dict[str, pd.DataFrame] = {}

    for table_name, dataframe in tables.items():
        safe = _prepare_dataframe(dataframe)

        csv_path = (
            PROCESSED_DIR
            / f"{table_name}.csv"
        )

        safe.to_csv(
            csv_path,
            index=False,
        )

        prepared_tables[table_name] = safe

        print(
            f"Prepared {table_name}: "
            f"{len(safe):,} rows"
        )

    engine = get_engine()

    # engine.begin() creates a PostgreSQL transaction.
    # It commits when successful and rolls back on failure.
    with engine.begin() as connection:
        for table_name, safe in prepared_tables.items():
            safe.to_sql(
                name=table_name,
                con=connection,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )

            print(
                f"Created PostgreSQL table: "
                f"{table_name}"
            )

        _create_indexes(connection)

    print(
        "\nAll cleaned tables were saved "
        "successfully in PostgreSQL."
    )


# ==========================================================
# INDEXES
# ==========================================================

def _create_indexes(connection) -> None:
    """
    Create indexes on frequently searched columns.
    """

    index_queries = [
        """
        CREATE INDEX IF NOT EXISTS ix_orders_order_id
        ON orders (order_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_logs_order_id
        ON delivery_logs (order_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS ix_reviews_product_id
        ON reviews (product_id)
        """,
    ]

    for sql in index_queries:
        connection.execute(
            text(sql)
        )

    print(
        "PostgreSQL indexes created successfully."
    )


# ==========================================================
# QUERY DATABASE
# ==========================================================

def query_df(
    sql: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Execute a PostgreSQL query and return the result as a
    pandas DataFrame.
    """

    with get_engine().connect() as connection:
        return pd.read_sql_query(
            sql=text(sql),
            con=connection,
            params=params or {},
        )


# ==========================================================
# DATABASE CONNECTION TEST
# ==========================================================

def test_connection() -> bool:
    """
    Check whether Python can connect to PostgreSQL.
    """

    with get_engine().connect() as connection:
        result = connection.execute(
            text(
                """
                SELECT
                    current_database() AS database_name,
                    current_user AS database_user
                """
            )
        ).mappings().one()

    print(
        "PostgreSQL connection successful."
    )

    print(
        f"Database: {result['database_name']}"
    )

    print(
        f"User    : {result['database_user']}"
    )

    return True


# ==========================================================
# CREATE DATABASE TABLES FROM CLEANED DATA
# ==========================================================

if __name__ == "__main__":
    from .cleaning import clean_all
    from .config import RAW_DIR

    print(
        "Testing PostgreSQL connection..."
    )

    test_connection()

    print(
        "\nCleaning SmartLogix datasets..."
    )

    cleaned_tables = clean_all(
        RAW_DIR
    )

    print(
        "\nCreating PostgreSQL tables "
        "from cleaned datasets..."
    )

    persist_tables(
        cleaned_tables
    )

    table_summary = query_df(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    )

    print(
        "\nPostgreSQL database created successfully."
    )

    print(
        f"Tables created: "
        f"{len(table_summary)}"
    )

    for table_name in table_summary[
        "table_name"
    ]:
        row_count = int(
            query_df(
                f'SELECT COUNT(*) AS total '
                f'FROM "{table_name}"'
            ).iloc[0]["total"]
        )

        print(
            f"  - {table_name}: "
            f"{row_count:,} rows"
        )