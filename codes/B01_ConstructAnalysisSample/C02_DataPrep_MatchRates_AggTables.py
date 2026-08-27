"""Build aggregate inventor-match inputs for ``C02_MatchRatesToInventorData.py``.

The output contains exact counts for the fixed all-country, U.S., and non-U.S. scopes.
Distinct-user denominators are computed inside each scope before data are written, so the
public marimo notebook never needs user identifiers or a user-set representation.

Run from the project root with the Talent environment:

    conda run -n Talent python codes/B01_ConstructAnalysisSample/C02_DataPrep_MatchRates_AggTables.py
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
MIN_COUNTRY_CANDIDATE_SPELLS = 1_000
REQUIRED_COLUMNS = (
    "user_id",
    "country",
    "state",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
    "seniority",
)


def hierarchy_number(column_name: str) -> int:
    match = re.search(r"_k(\d+)$", column_name)
    return int(match.group(1)) if match else -1


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def write_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
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
    """Repeat rows into the three fixed scopes required by the notebook."""

    return f"""
        SELECT 'all' AS scope_key, {value_expressions} FROM fnh
        UNION ALL
        SELECT 'us' AS scope_key, {value_expressions} FROM fnh
        WHERE country = {quote_literal(US_LABEL)}
        UNION ALL
        SELECT 'non_us' AS scope_key, {value_expressions} FROM fnh
        WHERE country <> {quote_literal(US_LABEL)}
          AND country <> {quote_literal(MISSING_LABEL)}
    """


def rate_aggregations() -> str:
    return """
        COUNT(*)::BIGINT AS candidate_spells,
        COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_spells,
        COUNT(DISTINCT user_id)::BIGINT AS unique_users,
        COUNT(DISTINCT user_id) FILTER (WHERE inventor_match)::BIGINT AS matched_users
    """


def classification_query(variable: str, title_column: str | None) -> str:
    value = quote_identifier(variable)
    title = (
        f"{quote_identifier(title_column)} AS title"
        if title_column
        else "CAST(NULL AS VARCHAR) AS title"
    )
    rows = scoped_rows(f"user_id, inventor_match, {value} AS value, {title}")
    return f"""
        SELECT scope_key, {quote_literal(variable)} AS variable, value, title,
               {rate_aggregations()}
        FROM ({rows})
        GROUP BY scope_key, value, title
        ORDER BY scope_key, candidate_spells DESC, value
    """


def joint_query(
    industry_column: str,
    occupation_column: str,
    title_columns: dict[str, str],
) -> str:
    industry = quote_identifier(industry_column)
    occupation = quote_identifier(occupation_column)
    industry_title_column = title_columns.get(industry_column)
    occupation_title_column = title_columns.get(occupation_column)
    industry_title = (
        f"{quote_identifier(industry_title_column)} AS industry_title"
        if industry_title_column
        else "CAST(NULL AS VARCHAR) AS industry_title"
    )
    occupation_title = (
        f"{quote_identifier(occupation_title_column)} AS occupation_title"
        if occupation_title_column
        else "CAST(NULL AS VARCHAR) AS occupation_title"
    )
    values = (
        f"user_id, inventor_match, {industry} AS industry_value, {industry_title}, "
        f"{occupation} AS occupation_value, {occupation_title}"
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
               {rate_aggregations()}
        FROM scoped
        WHERE industry_value IN (SELECT industry_value FROM eligible_industries)
        GROUP BY scope_key, industry_value, industry_title,
                 occupation_value, occupation_title
        ORDER BY scope_key, candidate_spells DESC, industry_value, occupation_value
    """


def prepare_staging_table(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
    crosswalk_path: Path,
    role_columns: tuple[str, ...],
    rics_columns: tuple[str, ...],
    date_column: str,
    row_limit: int | None,
) -> pd.DataFrame:
    """Materialize normalized focal hires and return crosswalk diagnostics."""

    crosswalk = quote_literal(crosswalk_path.as_posix())
    connection.execute(
        f"""
            CREATE TABLE patent_links AS
            SELECT TRY_CAST(user_id AS BIGINT) AS user_id,
                   TRIM(CAST(pv_inventor_id AS VARCHAR)) AS pv_inventor_id
            FROM read_csv_auto({crosswalk}, header=true, all_varchar=true)
        """
    )
    connection.execute(
        """
            CREATE TABLE patent_users AS
            SELECT DISTINCT user_id
            FROM patent_links
            WHERE user_id IS NOT NULL
              AND pv_inventor_id IS NOT NULL
              AND pv_inventor_id <> ''
        """
    )
    link_diagnostics = connection.execute(
        """
            WITH valid AS (
                SELECT * FROM patent_links
                WHERE user_id IS NOT NULL
                  AND pv_inventor_id IS NOT NULL
                  AND pv_inventor_id <> ''
            ), user_counts AS (
                SELECT user_id, COUNT(*) AS row_count FROM valid GROUP BY user_id
            )
            SELECT
                (SELECT COUNT(*) FROM patent_links)::BIGINT AS "Crosswalk rows",
                (SELECT COUNT(*) FROM valid)::BIGINT AS "Rows with both IDs",
                (SELECT COUNT(DISTINCT user_id) FROM valid)::BIGINT AS "Unique linked users",
                (SELECT COUNT(DISTINCT pv_inventor_id) FROM valid)::BIGINT
                    AS "Unique inventor IDs",
                (SELECT COUNT(*) FROM user_counts WHERE row_count > 1)::BIGINT
                    AS "Users with multiple rows",
                COALESCE((SELECT MAX(row_count) FROM user_counts), 0)::BIGINT
                    AS "Maximum rows per user",
                (SELECT COUNT(*) FROM patent_links WHERE user_id IS NULL)::BIGINT
                    AS "Missing user IDs",
                (SELECT COUNT(*) FROM patent_links WHERE pv_inventor_id IS NULL)::BIGINT
                    AS "Missing inventor IDs",
                (SELECT COUNT(*) FROM patent_links WHERE pv_inventor_id = '')::BIGINT
                    AS "Blank inventor IDs"
        """
    ).fetchdf()

    text_columns = (
        "country",
        "state",
        "seniority",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        *role_columns,
        *rics_columns,
    )
    select_columns = [
        "CAST(source.user_id AS BIGINT) AS user_id",
        *(normalized_text(column) for column in text_columns),
        (
            f"DATE_TRUNC('month', TRY_CAST({quote_identifier(date_column)} AS TIMESTAMP)) "
            "AS start_month"
        ),
        "patent_users.user_id IS NOT NULL AS inventor_match",
    ]
    limit = f"LIMIT {int(row_limit)}" if row_limit else ""
    connection.execute(
        "CREATE TABLE fnh AS SELECT "
        + ", ".join(select_columns)
        + f" FROM read_parquet({quote_literal(source_glob)}) AS source "
        + "LEFT JOIN patent_users ON CAST(source.user_id AS BIGINT) = patent_users.user_id "
        + limit
    )
    return link_diagnostics


def build_aggregates(
    input_dir: Path,
    crosswalk_path: Path,
    output_dir: Path,
    row_limit: int | None = None,
) -> None:
    source_glob = (input_dir / "*.parquet").as_posix()
    if not tuple(input_dir.glob("*.parquet")):
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Inventor crosswalk not found: {crosswalk_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="c02_aggregate_", dir=output_dir) as work_dir:
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
        link_diagnostics = prepare_staging_table(
            connection,
            source_glob,
            crosswalk_path,
            role_columns,
            rics_columns,
            date_column,
            row_limit,
        )
        link_diagnostics.to_parquet(output_dir / "link_diagnostics.parquet", index=False)

        title_columns = {"onet_code": "onet_title", "naics_code": "naics_description"}
        occupation_columns = ("onet_code", *role_columns)
        industry_columns = ("naics_code", *rics_columns)
        for variable in (*occupation_columns, *industry_columns, "seniority"):
            write_query(
                connection,
                classification_query(variable, title_columns.get(variable)),
                output_dir / "rates" / f"{variable}.parquet",
            )

        start_month_rows = scoped_rows("user_id, inventor_match, start_month AS value")
        write_query(
            connection,
            f"""
                SELECT scope_key, 'start_month' AS variable, value,
                       CAST(NULL AS VARCHAR) AS title, {rate_aggregations()}
                FROM ({start_month_rows})
                WHERE value IS NOT NULL
                GROUP BY scope_key, value
                ORDER BY scope_key, value
            """,
            output_dir / "rates" / "start_month.parquet",
        )

        for industry_column in industry_columns:
            for occupation_column in occupation_columns:
                filename = f"{industry_column}__{occupation_column}.parquet"
                write_query(
                    connection,
                    joint_query(industry_column, occupation_column, title_columns),
                    output_dir / "joint" / filename,
                )

        write_query(
            connection,
            f"""
                SELECT country AS value, {rate_aggregations()}
                FROM fnh
                WHERE country <> {quote_literal(MISSING_LABEL)}
                GROUP BY country
                ORDER BY candidate_spells DESC, value
            """,
            output_dir / "country_rates.parquet",
        )
        write_query(
            connection,
            f"""
                SELECT state AS value, {rate_aggregations()}
                FROM fnh
                WHERE country = {quote_literal(US_LABEL)}
                GROUP BY state
                ORDER BY candidate_spells DESC, value
            """,
            output_dir / "us_state_rates.parquet",
        )

        scope_rows = scoped_rows("user_id, inventor_match")
        write_query(
            connection,
            f"""
                SELECT scope_key, {rate_aggregations()}
                FROM ({scope_rows}) GROUP BY scope_key
                ORDER BY CASE scope_key WHEN 'all' THEN 1 WHEN 'us' THEN 2 ELSE 3 END
            """,
            output_dir / "scope_totals.parquet",
        )

        metadata = connection.execute(
            f"""
                SELECT COUNT(*)::BIGINT AS candidate_count,
                       COUNT(DISTINCT user_id)::BIGINT AS distinct_users,
                       COUNT(DISTINCT NULLIF(country, {quote_literal(MISSING_LABEL)}))::BIGINT
                           AS nonmissing_country_count
                FROM fnh
            """
        ).fetchdf()
        metadata["aggregate_schema_version"] = 4
        metadata["source_notebook"] = "B03_FNH_InventorMatchRates.py"
        metadata["notebook_input_level"] = "aggregate_counts"
        metadata["available_role_columns"] = "|".join(role_columns)
        metadata["available_rics_columns"] = "|".join(rics_columns)
        metadata["available_occupation_columns"] = "|".join(occupation_columns)
        metadata["available_industry_columns"] = "|".join(industry_columns)
        metadata["country_scope_keys"] = "all|us|non_us"
        metadata["minimum_country_candidate_spells"] = MIN_COUNTRY_CANDIDATE_SPELLS
        metadata["minimum_all_country_industry_hires"] = MIN_ALL_COUNTRY_INDUSTRY_HIRES
        metadata.to_parquet(output_dir / "metadata.parquet", index=False)
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
        "--crosswalk-path",
        type=Path,
        default=(
            project_root
            / "data"
            / "a_raw_data"
            / "A_Revelio"
            / "revelio_user_id_patentsview_id.csv"
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
            / "C02_MatchRatesToInventorData"
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
    build_aggregates(
        arguments.input_dir,
        arguments.crosswalk_path,
        arguments.output_dir,
        arguments.row_limit,
    )
