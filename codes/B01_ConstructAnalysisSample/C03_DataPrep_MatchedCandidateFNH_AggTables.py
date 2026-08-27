"""Build aggregate inputs for ``C03_MatchedCandidateFNH.py``.

Counts are stored by country so the companion notebook can retain its flexible country
controls without retaining user IDs.  Matched and universe counts are written together,
which makes every displayed comparison use the same category support.

Run from the project root with the Talent environment:

    conda run -n Talent python codes/B01_ConstructAnalysisSample/C03_DataPrep_MatchedCandidateFNH_AggTables.py
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


def classification_query(variable: str, title_column: str | None) -> str:
    value = quote_identifier(variable)
    title = (
        f"{quote_identifier(title_column)} AS title"
        if title_column
        else "CAST(NULL AS VARCHAR) AS title"
    )
    return f"""
        SELECT country, {quote_literal(variable)} AS variable,
               {value} AS value, {title},
               COUNT(*)::BIGINT AS universe_count,
               COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count
        FROM fnh
        GROUP BY country, value, title
        ORDER BY country, universe_count DESC, value
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
    return f"""
        WITH eligible_industries AS (
            SELECT {industry} AS industry_value
            FROM fnh
            WHERE {industry} <> {quote_literal(MISSING_LABEL)}
            GROUP BY {industry}
            HAVING COUNT(*) >= {MIN_ALL_COUNTRY_INDUSTRY_HIRES}
        )
        SELECT country,
               {quote_literal(industry_column)} AS industry_variable,
               {quote_literal(occupation_column)} AS occupation_variable,
               {industry} AS industry_value, {industry_title},
               {occupation} AS occupation_value, {occupation_title},
               COUNT(*)::BIGINT AS universe_count,
               COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count
        FROM fnh
        WHERE {industry} IN (SELECT industry_value FROM eligible_industries)
        GROUP BY country, industry_value, industry_title,
                 occupation_value, occupation_title
        ORDER BY country, universe_count DESC, industry_value, occupation_value
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
                    AS "Maximum rows per user"
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
        "CAST(source.rcid AS VARCHAR) AS rcid",
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


def schema_report(
    connection: duckdb.DuckDBPyConnection,
    available_columns: tuple[str, ...],
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    matched_count = connection.execute("SELECT COUNT(*) FROM fnh WHERE inventor_match").fetchone()[
        0
    ]
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
            f"SELECT COUNT(*) FROM fnh WHERE inventor_match AND "
            f"{quote_identifier(column)} = {quote_literal(MISSING_LABEL)}"
        ).fetchone()[0]
        rows.append(
            {
                "Variable": column,
                "Status": "Available",
                "Missing rows": int(missing_count),
                "Missing share": (missing_count / matched_count if matched_count else float("nan")),
            }
        )
    return pd.DataFrame(rows)


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

    with tempfile.TemporaryDirectory(prefix="c03_aggregate_", dir=output_dir) as work_dir:
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
        classification_columns = (*occupation_columns, *industry_columns, "seniority")
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

        write_query(
            connection,
            """
                SELECT country,
                       COUNT(*)::BIGINT AS universe_count,
                       COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count
                FROM fnh GROUP BY country ORDER BY universe_count DESC, country
            """,
            output_dir / "country_counts.parquet",
        )
        write_query(
            connection,
            f"""
                SELECT state,
                       COUNT(*)::BIGINT AS universe_count,
                       COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count
                FROM fnh WHERE country = {quote_literal(US_LABEL)}
                GROUP BY state ORDER BY universe_count DESC, state
            """,
            output_dir / "us_state_counts.parquet",
        )
        write_query(
            connection,
            """
                SELECT country, start_month,
                       COUNT(*)::BIGINT AS universe_count,
                       COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count
                FROM fnh WHERE start_month IS NOT NULL
                GROUP BY country, start_month ORDER BY country, start_month
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
        schema_report(connection, available_columns, expected_columns).to_parquet(
            output_dir / "schema_report.parquet", index=False
        )

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
                    WHERE inventor_match
                      AND {quote_identifier(code_column)} <> {quote_literal(MISSING_LABEL)}
                      AND {quote_identifier(title_column)} <> {quote_literal(MISSING_LABEL)}
                    GROUP BY {quote_identifier(code_column)}
                """
            ).fetchdf()
            diagnostic.insert(0, "classification", label)
            diagnostics.append(diagnostic)
        pd.concat(diagnostics, ignore_index=True).to_parquet(
            output_dir / "title_diagnostics.parquet", index=False
        )

        metadata = connection.execute(
            f"""
                SELECT
                    COUNT(*) FILTER (WHERE inventor_match)::BIGINT AS matched_count,
                    COUNT(*)::BIGINT AS universe_count,
                    COUNT(DISTINCT user_id) FILTER (WHERE inventor_match)::BIGINT
                        AS matched_distinct_users,
                    COUNT(DISTINCT user_id)::BIGINT AS universe_distinct_users,
                    COUNT(DISTINCT rcid) FILTER (WHERE inventor_match)::BIGINT
                        AS matched_distinct_companies,
                    COUNT(DISTINCT rcid)::BIGINT AS universe_distinct_companies,
                    COUNT(DISTINCT NULLIF(country, {quote_literal(MISSING_LABEL)}))
                        FILTER (WHERE inventor_match)::BIGINT AS matched_distinct_countries,
                    COUNT(DISTINCT NULLIF(country, {quote_literal(MISSING_LABEL)}))::BIGINT
                        AS universe_distinct_countries
                FROM fnh
            """
        ).fetchdf()
        metadata["aggregate_schema_version"] = 3
        metadata["source_notebook"] = "B04_FNH_MatchedSummaryStats.py"
        metadata["notebook_input_level"] = "aggregate_counts"
        metadata["available_role_columns"] = "|".join(role_columns)
        metadata["available_rics_columns"] = "|".join(rics_columns)
        metadata["available_occupation_columns"] = "|".join(occupation_columns)
        metadata["available_industry_columns"] = "|".join(industry_columns)
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
            / "C03_MatchedCandidateFNH"
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
