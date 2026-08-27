"""Build privacy-safe aggregate inputs for ``C01_UniverseCandidateFNH.py``.

The source data contain one row per user-company spell.  This script is the only
aggregate-version component that reads those rows.  The companion marimo notebook reads
only the compressed Parquet tables written here.

Run from the project root with the Talent environment:

    conda run -n Talent python codes/B01_ConstructAnalysisSample/C01_DataPrep_UniverseCandidateFNH_AggTables.py
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

MISSING_LABEL = "<Missing>"
US_LABEL = "United States"
MIN_ALL_COUNTRY_INDUSTRY_HIRES = 1_000
EXPECTED_ROLE_COLUMNS = (
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1500",
)
EXPECTED_RICS_COLUMNS = ("rics_k50", "rics_k200", "rics_k400")
REQUIRED_COLUMNS = (
    "user_id",
    "rcid",
    "country",
    "state",
    "seniority",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
)


def hierarchy_number(column_name: str) -> int:
    """Return the numeric level in a Revelio hierarchy column name."""

    match = re.search(r"_k(\d+)$", column_name)
    return int(match.group(1)) if match else -1


def quote_identifier(value: str) -> str:
    """Quote a trusted schema identifier for DuckDB."""

    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str | Path) -> str:
    """Quote a string or path as a DuckDB SQL literal."""

    return "'" + str(value).replace("'", "''") + "'"


def parquet_glob(input_dir: Path) -> str:
    return (input_dir / "*.parquet").as_posix()


def write_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    """Write a query to compressed Parquet, replacing the named output atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    connection.execute(
        f"COPY ({query}) TO {quote_literal(temporary_path.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    temporary_path.replace(path)


def source_columns(connection: duckdb.DuckDBPyConnection, source_glob: str) -> tuple[str, ...]:
    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_literal(source_glob)})"
    ).fetchdf()
    return tuple(description["column_name"].astype(str))


def normalized_text(column: str) -> str:
    identifier = quote_identifier(column)
    return (
        f"COALESCE(NULLIF(TRIM(CAST({identifier} AS VARCHAR)), ''), "
        f"{quote_literal(MISSING_LABEL)}) AS {identifier}"
    )


def scoped_rows(value_expressions: str) -> str:
    """Return all-country and U.S. rows for one aggregate query."""

    return f"""
        SELECT 'all' AS scope_key, {value_expressions}
        FROM fnh
        UNION ALL
        SELECT 'us' AS scope_key, {value_expressions}
        FROM fnh
        WHERE country = {quote_literal(US_LABEL)}
    """


def classification_query(variable: str, title_column: str | None) -> str:
    value = quote_identifier(variable)
    title = (
        f"{quote_identifier(title_column)} AS title"
        if title_column
        else "CAST(NULL AS VARCHAR) AS title"
    )
    rows = scoped_rows(f"{value} AS value, {title}")
    return f"""
        SELECT scope_key, {quote_literal(variable)} AS variable, value, title,
               COUNT(*)::BIGINT AS count
        FROM ({rows})
        GROUP BY scope_key, value, title
        ORDER BY scope_key, count DESC, value
    """


def joint_query(
    industry_column: str,
    occupation_column: str,
    title_columns: dict[str, str],
) -> str:
    industry = quote_identifier(industry_column)
    occupation = quote_identifier(occupation_column)
    industry_title = title_columns.get(industry_column)
    occupation_title = title_columns.get(occupation_column)
    industry_title_sql = (
        f"{quote_identifier(industry_title)} AS industry_title"
        if industry_title
        else "CAST(NULL AS VARCHAR) AS industry_title"
    )
    occupation_title_sql = (
        f"{quote_identifier(occupation_title)} AS occupation_title"
        if occupation_title
        else "CAST(NULL AS VARCHAR) AS occupation_title"
    )
    values = (
        f"{industry} AS industry_value, {industry_title_sql}, "
        f"{occupation} AS occupation_value, {occupation_title_sql}"
    )
    rows = scoped_rows(values)
    return f"""
        WITH eligible_industries AS (
            SELECT {industry} AS industry_value
            FROM fnh
            WHERE {industry} <> {quote_literal(MISSING_LABEL)}
            GROUP BY {industry}
            HAVING COUNT(*) >= {MIN_ALL_COUNTRY_INDUSTRY_HIRES}
        ), scoped AS ({rows})
        SELECT scope_key,
               {quote_literal(industry_column)} AS industry_variable,
               {quote_literal(occupation_column)} AS occupation_variable,
               industry_value, industry_title, occupation_value, occupation_title,
               COUNT(*)::BIGINT AS count
        FROM scoped
        WHERE industry_value IN (SELECT industry_value FROM eligible_industries)
        GROUP BY scope_key, industry_value, industry_title,
                 occupation_value, occupation_title
        ORDER BY scope_key, count DESC, industry_value, occupation_value
    """


def crosswalk_query(
    left_column: str,
    right_column: str,
    title_columns: dict[str, str],
) -> str:
    left = quote_identifier(left_column)
    right = quote_identifier(right_column)
    left_title_column = title_columns.get(left_column)
    right_title_column = title_columns.get(right_column)
    left_title = (
        f"{quote_identifier(left_title_column)} AS left_title"
        if left_title_column
        else "CAST(NULL AS VARCHAR) AS left_title"
    )
    right_title = (
        f"{quote_identifier(right_title_column)} AS right_title"
        if right_title_column
        else "CAST(NULL AS VARCHAR) AS right_title"
    )
    rows = scoped_rows(f"{left} AS left_value, {left_title}, {right} AS right_value, {right_title}")
    return f"""
        SELECT scope_key,
               {quote_literal(left_column)} AS left_variable,
               {quote_literal(right_column)} AS right_variable,
               left_value, left_title, right_value, right_title,
               COUNT(*)::BIGINT AS count
        FROM ({rows})
        GROUP BY scope_key, left_value, left_title, right_value, right_title
        ORDER BY scope_key, left_value, count DESC, right_value
    """


def schema_report(
    connection: duckdb.DuckDBPyConnection,
    available_columns: tuple[str, ...],
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    row_count = connection.execute("SELECT COUNT(*) FROM fnh").fetchone()[0]
    rows = []
    for column in expected_columns:
        if column not in available_columns:
            rows.append(
                {
                    "Variable": column,
                    "Status": "Absent from input schema",
                    "Missing rows": pd.NA,
                    "Missing share": pd.NA,
                }
            )
            continue
        missing_count = connection.execute(
            f"SELECT COUNT(*) FROM fnh WHERE {quote_identifier(column)} = "
            f"{quote_literal(MISSING_LABEL)}"
        ).fetchone()[0]
        rows.append(
            {
                "Variable": column,
                "Status": "Available",
                "Missing rows": int(missing_count),
                "Missing share": missing_count / row_count if row_count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def prepare_staging_table(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
    available_role_columns: tuple[str, ...],
    available_rics_columns: tuple[str, ...],
    date_column: str,
    row_limit: int | None,
) -> None:
    text_columns = (
        "country",
        "state",
        "seniority",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        *available_role_columns,
        *available_rics_columns,
    )
    select_columns = [
        "CAST(user_id AS BIGINT) AS user_id",
        "CAST(rcid AS VARCHAR) AS rcid",
        *(normalized_text(column) for column in text_columns),
        (
            f"DATE_TRUNC('month', TRY_CAST({quote_identifier(date_column)} AS TIMESTAMP)) "
            "AS start_month"
        ),
    ]
    limit = f"LIMIT {int(row_limit)}" if row_limit else ""
    connection.execute(
        "CREATE TABLE fnh AS SELECT "
        + ", ".join(select_columns)
        + f" FROM read_parquet({quote_literal(source_glob)}) {limit}"
    )


def build_aggregates(input_dir: Path, output_dir: Path, row_limit: int | None = None) -> None:
    """Build all compact aggregate tables used by the C01 marimo notebook."""

    source_glob = parquet_glob(input_dir)
    if not tuple(input_dir.glob("*.parquet")):
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="c01_aggregate_", dir=output_dir) as work_dir:
        connection = duckdb.connect(str(Path(work_dir) / "work.duckdb"))
        available_columns = source_columns(connection, source_glob)
        missing_required = sorted(set(REQUIRED_COLUMNS) - set(available_columns))
        if missing_required:
            raise ValueError(f"Input is missing required fields: {missing_required}")
        if "start_month" in available_columns:
            date_column = "start_month"
        elif "startdate" in available_columns:
            date_column = "startdate"
        else:
            raise ValueError("Input must contain either `start_month` or `startdate`.")

        role_columns = tuple(
            sorted(
                (column for column in available_columns if column.startswith("role_k")),
                key=hierarchy_number,
            )
        )
        rics_columns = tuple(
            sorted(
                (column for column in available_columns if column.startswith("rics_k")),
                key=hierarchy_number,
            )
        )
        prepare_staging_table(
            connection,
            source_glob,
            role_columns,
            rics_columns,
            date_column,
            row_limit,
        )

        title_columns = {"onet_code": "onet_title", "naics_code": "naics_description"}
        classification_columns = (
            "onet_code",
            *role_columns,
            "naics_code",
            *rics_columns,
        )
        industry_columns = ("naics_code", *rics_columns)
        occupation_columns = ("onet_code", *role_columns)

        for variable in classification_columns:
            write_query(
                connection,
                classification_query(variable, title_columns.get(variable)),
                output_dir / "classification" / f"{variable}.parquet",
            )

        for industry_column in industry_columns:
            for occupation_column in occupation_columns:
                filename = f"{industry_column}__{occupation_column}.parquet"
                write_query(
                    connection,
                    joint_query(industry_column, occupation_column, title_columns),
                    output_dir / "joint" / filename,
                )

        for role_column in role_columns:
            filename = f"onet_code__{role_column}.parquet"
            write_query(
                connection,
                crosswalk_query("onet_code", role_column, title_columns),
                output_dir / "crosswalk" / filename,
            )

        finest_rics_column = rics_columns[-1] if rics_columns else None
        if finest_rics_column:
            for left_column, right_column in (
                ("naics_code", finest_rics_column),
                (finest_rics_column, "naics_code"),
            ):
                filename = f"{left_column}__{right_column}.parquet"
                write_query(
                    connection,
                    crosswalk_query(left_column, right_column, title_columns),
                    output_dir / "crosswalk" / filename,
                )

        role_crosswalk_columns = (*role_columns, "onet_code", "onet_title")
        role_crosswalk_select = ", ".join(map(quote_identifier, role_crosswalk_columns))
        if "role_k1500" in role_columns:
            role_crosswalk_query = f"""
                WITH shares AS (
                    SELECT role_k1500, COUNT(*)::BIGINT AS role_count,
                           COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () AS role_share
                    FROM fnh
                    GROUP BY role_k1500
                )
                SELECT DISTINCT {role_crosswalk_select}, shares.role_count, shares.role_share
                FROM fnh
                LEFT JOIN shares USING (role_k1500)
                ORDER BY role_share DESC, role_k1500, onet_code, onet_title
            """
        else:
            role_crosswalk_query = f"SELECT DISTINCT {role_crosswalk_select} FROM fnh"
        write_query(
            connection,
            role_crosswalk_query,
            output_dir / "role_onet_crosswalk.parquet",
        )

        write_query(
            connection,
            """
                SELECT country, COUNT(*)::BIGINT AS count
                FROM fnh GROUP BY country ORDER BY count DESC, country
            """,
            output_dir / "country_counts.parquet",
        )
        write_query(
            connection,
            f"""
                SELECT state, COUNT(*)::BIGINT AS count
                FROM fnh WHERE country = {quote_literal(US_LABEL)}
                GROUP BY state ORDER BY count DESC, state
            """,
            output_dir / "us_state_counts.parquet",
        )
        write_query(
            connection,
            f"""
                WITH scoped AS ({scoped_rows("seniority AS value")})
                SELECT scope_key, value, COUNT(*)::BIGINT AS count
                FROM scoped GROUP BY scope_key, value
                ORDER BY scope_key, count DESC, value
            """,
            output_dir / "seniority_counts.parquet",
        )
        write_query(
            connection,
            f"""
                WITH scoped AS ({scoped_rows("start_month")})
                SELECT scope_key, start_month, COUNT(*)::BIGINT AS count
                FROM scoped WHERE start_month IS NOT NULL
                GROUP BY scope_key, start_month ORDER BY scope_key, start_month
            """,
            output_dir / "time_counts.parquet",
        )

        expected_columns = (
            "onet_code",
            "onet_title",
            *EXPECTED_ROLE_COLUMNS,
            *EXPECTED_RICS_COLUMNS,
            "naics_code",
            "naics_description",
            "country",
            "state",
            "seniority",
        )
        schema = schema_report(connection, available_columns, expected_columns)
        schema.to_parquet(output_dir / "schema_report.parquet", index=False)

        diagnostics = []
        for code_column, title_column, label in (
            ("onet_code", "onet_title", "ONET"),
            ("naics_code", "naics_description", "NAICS"),
        ):
            diagnostic = connection.execute(
                f"""
                    SELECT {quote_identifier(code_column)} AS code,
                           COUNT(DISTINCT {quote_identifier(title_column)})::BIGINT
                               AS distinct_titles
                    FROM fnh
                    WHERE {quote_identifier(code_column)} <> {quote_literal(MISSING_LABEL)}
                      AND {quote_identifier(title_column)} <> {quote_literal(MISSING_LABEL)}
                    GROUP BY {quote_identifier(code_column)}
                """
            ).fetchdf()
            diagnostic.insert(0, "classification", label)
            diagnostics.append(diagnostic)
        pd.concat(diagnostics, ignore_index=True).to_parquet(
            output_dir / "title_diagnostics.parquet", index=False
        )

        totals = connection.execute(
            f"""
                SELECT COUNT(*)::BIGINT AS candidate_count,
                       COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
                       COUNT(DISTINCT rcid)::BIGINT AS distinct_companies,
                       COUNT(DISTINCT NULLIF(country, {quote_literal(MISSING_LABEL)}))::BIGINT
                           AS distinct_countries
                FROM fnh
            """
        ).fetchdf()
        totals["aggregate_schema_version"] = 3
        totals["source_notebook"] = "B02_FNH_SummaryStats.py"
        totals["notebook_input_level"] = "aggregate_counts"
        totals["available_role_columns"] = "|".join(role_columns)
        totals["available_rics_columns"] = "|".join(rics_columns)
        totals["finest_rics_column"] = finest_rics_column or ""
        totals["country_scope_keys"] = "all|us"
        totals["minimum_all_country_industry_hires"] = MIN_ALL_COUNTRY_INDUSTRY_HIRES
        totals.to_parquet(output_dir / "metadata.parquet", index=False)
        connection.close()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            project_root
            / "data"
            / "b_temp_data"
            / "B01_ConstructAnalysisSample"
            / "FocalNewHires_AllIndustries"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "data"
            / "b_temp_data"
            / "B01_ConstructAnalysisSample"
            / "C01_UniverseCandidateFNH"
        ),
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional development-only row limit; omit for production aggregates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_aggregates(arguments.input_dir, arguments.output_dir, arguments.row_limit)
