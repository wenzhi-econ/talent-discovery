# ruff: noqa: N999

"""Build aggregate-only inputs for the public USPTO-inventor marimo report.

The source extract contains one row per inventor employment spell. This script is the
only component of the public report that reads those rows. The companion notebook reads
only the compressed Parquet tables written here.

Run from the project root with the Talent environment:

    conda run -s -n Talent python codes/B02_InspectUSPTOInventors/E03_CreateAggTables.py
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
    "position_id",
    "rcid",
    "country",
    "seniority",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
    "title_raw",
)
TITLE_COLUMNS = {"onet_code": "onet_title", "naics_code": "naics_description"}
TITLE_ABBREVIATION_REPLACEMENTS = (
    (r"\bsnr\b", "senior"),
    (r"\bsr\b", "senior"),
    (r"\bjr\b", "junior"),
    (r"\basst\b", "assistant"),
    (r"\bassoc\b", "associate"),
    (r"\bcoord\b", "coordinator"),
    (r"\bdir\b", "director"),
    (r"\bengr\b", "engineer"),
    (r"\bexec\b", "executive"),
    (r"\bmngr\b", "manager"),
    (r"\bmgr\b", "manager"),
    (r"\bsupv\b", "supervisor"),
    (r"\bdept\b", "department"),
    (r"\bintl\b", "international"),
    (r"\bmfg\b", "manufacturing"),
    (r"\bmktg\b", "marketing"),
    (r"\bmgmt\b", "management"),
    (r"\bops\b", "operations"),
    (r"\bsvp\b", "senior vice president"),
    (r"\bevp\b", "executive vice president"),
    (r"\bavp\b", "assistant vice president"),
    (r"\bvp\b", "vice president"),
    (r"\bceo\b", "chief executive officer"),
    (r"\bcfo\b", "chief financial officer"),
    (r"\bchro\b", "chief human resources officer"),
    (r"\bcio\b", "chief information officer"),
    (r"\bcoo\b", "chief operating officer"),
    (r"\bcto\b", "chief technology officer"),
    (r"\bhr\b", "human resources"),
    (r"\bqa\b", "quality assurance"),
    (r"\bqc\b", "quality control"),
    (r"\bcofounder\b", "co founder"),
)


def hierarchy_number(column_name: str) -> int:
    """Return the numeric resolution in a Revelio hierarchy column name."""

    match = re.search(r"_k(\d+)$", column_name)
    return int(match.group(1)) if match else -1


def quote_identifier(value: str) -> str:
    """Quote a trusted schema identifier for DuckDB."""

    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str | Path) -> str:
    """Quote a string or path as a DuckDB SQL literal."""

    return "'" + str(value).replace("'", "''") + "'"


def parquet_glob(input_dir: Path) -> str:
    """Return DuckDB's platform-neutral Parquet glob for the source directory."""

    return (input_dir / "*.parquet").as_posix()


def write_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    """Write a query to compressed Parquet, replacing one named output atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    connection.execute(
        f"COPY ({query}) TO {quote_literal(temporary_path.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    temporary_path.replace(path)


def source_columns(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
) -> tuple[str, ...]:
    """Read the input schema without materializing the source rows."""

    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_literal(source_glob)})"
    ).fetchdf()
    return tuple(description["column_name"].astype(str))


def normalize_job_titles(job_titles: pd.Series) -> pd.Series:
    """Apply the exact title-normalization rules used by ``E02_USPTOInventors.py``."""

    normalized_titles = (
        job_titles.astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&amp;", "&", regex=False)
        .str.strip()
    )
    normalized_titles = normalized_titles.str.replace(
        r"\br\s*(?:&|/|\+)\s*d\b",
        "research and development",
        regex=True,
    )
    normalized_titles = normalized_titles.str.replace(
        r"\br\s+and\s+d\b",
        "research and development",
        regex=True,
    )
    normalized_titles = normalized_titles.str.replace(r"\s+\+\s+", " and ", regex=True)
    normalized_titles = normalized_titles.str.replace("&", " and ", regex=False)
    normalized_titles = normalized_titles.str.replace(".", "", regex=False)
    normalized_titles = normalized_titles.str.replace(
        "['\u2018\u2019\u02bc]", "", regex=True
    )
    normalized_titles = normalized_titles.str.replace(
        "[-\u2010-\u2015\u2212/|,;:_()\\[\\]{}]+",
        " ",
        regex=True,
    )
    normalized_titles = normalized_titles.str.replace('"', " ", regex=False)
    normalized_titles = normalized_titles.str.replace("\\", " ", regex=False)
    normalized_titles = normalized_titles.str.replace(
        r"[!?@%^*=<>~`$]+", " ", regex=True
    )
    normalized_titles = normalized_titles.str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    for pattern, replacement in TITLE_ABBREVIATION_REPLACEMENTS:
        normalized_titles = normalized_titles.str.replace(
            pattern, replacement, regex=True
        )
    normalized_titles = normalized_titles.str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    return normalized_titles.mask(normalized_titles.eq(""), pd.NA)


def prepare_staging_table(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
    role_columns: tuple[str, ...],
    rics_columns: tuple[str, ...],
    row_limit: int | None,
) -> None:
    """Materialize the retained columns once for the many downstream group-bys."""

    text_columns = (
        "country",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        *role_columns,
        *rics_columns,
        "title_raw",
    )
    text_select = [
        f"CAST({quote_identifier(column)} AS VARCHAR) AS {quote_identifier(column)}"
        for column in text_columns
    ]
    limit = f"LIMIT {int(row_limit)}" if row_limit else ""
    source_query = (
        "SELECT * FROM read_parquet(" + quote_literal(source_glob) + f") {limit}"
    )
    select_columns = [
        "ROW_NUMBER() OVER ()::BIGINT AS row_order",
        "CAST(user_id AS BIGINT) AS user_id",
        "CAST(position_id AS BIGINT) AS position_id",
        "CAST(rcid AS BIGINT) AS rcid",
        *text_select,
        "CAST(seniority AS SMALLINT) AS seniority",
    ]
    connection.execute(
        "CREATE TABLE spells AS SELECT "
        + ", ".join(select_columns)
        + f" FROM ({source_query}) AS source WHERE rcid IS NOT NULL"
    )
    connection.execute("UPDATE spells SET country = NULL WHERE country = 'empty'")
    connection.execute("ALTER TABLE spells ADD COLUMN is_us BOOLEAN DEFAULT FALSE")
    connection.execute(
        f"UPDATE spells SET is_us = COALESCE(country = {quote_literal(US_LABEL)}, FALSE)"
    )


def label_expression(variable: str, title_column: str | None) -> tuple[str, str]:
    """Return raw-value and optional title expressions matching the notebook labels."""

    value = (
        f"COALESCE(CAST({quote_identifier(variable)} AS VARCHAR), "
        f"{quote_literal(MISSING_LABEL)})"
    )
    title = quote_identifier(title_column) if title_column else "CAST(NULL AS VARCHAR)"
    return value, title


def classification_query(variable: str, title_column: str | None) -> str:
    """Aggregate a classification by U.S. status and seniority."""

    value, title = label_expression(variable, title_column)
    return f"""
        SELECT is_us, seniority, {quote_literal(variable)} AS variable,
               {value} AS value, {title} AS title,
               COUNT(*)::BIGINT AS counts,
               MIN(row_order)::BIGINT AS first_order
        FROM spells
        GROUP BY is_us, seniority, value, title
        ORDER BY is_us, seniority, counts DESC, first_order
    """


def joint_query(industry_column: str, occupation_column: str) -> str:
    """Aggregate one selectable industry-occupation pairing."""

    industry_value, _ = label_expression(industry_column, None)
    occupation_value, _ = label_expression(occupation_column, None)
    return f"""
        SELECT is_us, seniority,
               {quote_literal(industry_column)} AS industry_variable,
               {quote_literal(occupation_column)} AS occupation_variable,
               {industry_value} AS industry_value,
               {occupation_value} AS occupation_value,
               COUNT(*)::BIGINT AS counts
        FROM spells
        GROUP BY is_us, seniority, industry_value, occupation_value
        ORDER BY is_us, seniority, counts DESC, industry_value, occupation_value
    """


def write_title_counts(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> None:
    """Write the restricted title cube without raw titles or person identifiers."""

    raw_titles = connection.execute(
        """
        SELECT title_raw, MIN(row_order)::BIGINT AS first_order
        FROM spells
        WHERE is_us AND seniority < 5 AND title_raw IS NOT NULL
        GROUP BY title_raw
        ORDER BY first_order
        """
    ).fetchdf()
    raw_titles["title_normalized"] = normalize_job_titles(raw_titles["title_raw"])
    raw_titles["raw_title_variant_id"] = pd.Series(
        range(1, len(raw_titles) + 1),
        dtype="int64",
    )
    title_lookup = raw_titles.loc[
        :, ["title_raw", "title_normalized", "raw_title_variant_id"]
    ]
    connection.register("title_lookup_frame", title_lookup)
    connection.execute(
        "CREATE TEMP TABLE title_lookup AS SELECT * FROM title_lookup_frame"
    )
    connection.unregister("title_lookup_frame")

    industry_value, _ = label_expression("rics_k400", None)
    occupation_value, _ = label_expression("onet_code", None)
    write_query(
        connection,
        f"""
        SELECT spells.seniority,
               {industry_value} AS industry_value,
               {occupation_value} AS occupation_value,
               title_lookup.title_normalized,
               title_lookup.raw_title_variant_id,
               COUNT(*)::BIGINT AS counts
        FROM spells
        LEFT JOIN title_lookup USING (title_raw)
        WHERE spells.is_us AND spells.seniority < 5
        GROUP BY spells.seniority, industry_value, occupation_value,
                 title_lookup.title_normalized, title_lookup.raw_title_variant_id
        ORDER BY spells.seniority, industry_value, occupation_value,
                 counts DESC, title_lookup.title_normalized
        """,
        output_dir / "title_counts.parquet",
    )


def build_aggregates(
    input_dir: Path,
    output_dir: Path,
    row_limit: int | None = None,
) -> None:
    """Build every aggregate table required by the public marimo notebook."""

    source_glob = parquet_glob(input_dir)
    parquet_files = tuple(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="e03_aggregate_", dir=output_dir
    ) as work_dir:
        connection = duckdb.connect(str(Path(work_dir) / "work.duckdb"))
        available_columns = source_columns(connection, source_glob)
        missing_required = sorted(set(REQUIRED_COLUMNS) - set(available_columns))
        if missing_required:
            raise ValueError(f"Input is missing required fields: {missing_required}")

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
        missing_expected = sorted(
            (set(EXPECTED_ROLE_COLUMNS) | set(EXPECTED_RICS_COLUMNS))
            - set(available_columns)
        )
        if missing_expected:
            raise ValueError(
                f"Input is missing expected hierarchy fields: {missing_expected}"
            )

        prepare_staging_table(
            connection,
            source_glob,
            role_columns,
            rics_columns,
            row_limit,
        )
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
                classification_query(variable, TITLE_COLUMNS.get(variable)),
                output_dir / "classification" / f"{variable}.parquet",
            )
        for industry_column in industry_columns:
            for occupation_column in occupation_columns:
                write_query(
                    connection,
                    joint_query(industry_column, occupation_column),
                    output_dir
                    / "joint"
                    / f"{industry_column}__{occupation_column}.parquet",
                )

        write_query(
            connection,
            """
            WITH scoped AS (
                SELECT 'all' AS scope_key, * FROM spells
                UNION ALL
                SELECT 'us' AS scope_key, * FROM spells WHERE is_us
            )
            SELECT scope_key,
                   COUNT(*)::BIGINT AS employment_spells,
                   COUNT(DISTINCT user_id)::BIGINT AS distinct_inventors,
                   COUNT(DISTINCT rcid)::BIGINT AS distinct_companies,
                   COUNT(DISTINCT country)::BIGINT AS distinct_countries,
                   COUNT(DISTINCT onet_code)::BIGINT AS distinct_onet_occupations,
                   COUNT(DISTINCT naics_code)::BIGINT AS distinct_naics_industries
            FROM scoped GROUP BY scope_key ORDER BY scope_key
            """,
            output_dir / "basic_numbers.parquet",
        )
        write_query(
            connection,
            """
            SELECT 'Employment-spell rows' AS diagnostic, COUNT(*)::BIGINT AS value
            FROM spells
            UNION ALL
            SELECT 'Distinct position_id values', COUNT(DISTINCT position_id)::BIGINT
            FROM spells
            UNION ALL
            SELECT 'Rows with a missing position_id',
                   COUNT(*) FILTER (WHERE position_id IS NULL)::BIGINT
            FROM spells
            UNION ALL
            SELECT 'Rows with a missing country',
                   COUNT(*) FILTER (WHERE country IS NULL)::BIGINT
            FROM spells
            UNION ALL
            SELECT 'Rows with a missing ONET occupation',
                   COUNT(*) FILTER (WHERE onet_code IS NULL)::BIGINT
            FROM spells
            UNION ALL
            SELECT 'Rows with a missing NAICS industry',
                   COUNT(*) FILTER (WHERE naics_code IS NULL)::BIGINT
            FROM spells
            """,
            output_dir / "quality_diagnostics.parquet",
        )
        write_query(
            connection,
            """
            SELECT is_us, seniority, COUNT(*)::BIGINT AS counts,
                   MIN(row_order)::BIGINT AS first_order
            FROM spells
            GROUP BY is_us, seniority
            ORDER BY is_us, counts DESC, first_order
            """,
            output_dir / "seniority_counts.parquet",
        )
        write_query(
            connection,
            f"""
            SELECT COALESCE(country, {quote_literal(MISSING_LABEL)}) AS value,
                   COUNT(*)::BIGINT AS counts,
                   MIN(row_order)::BIGINT AS first_order
            FROM spells
            GROUP BY value
            ORDER BY counts DESC, first_order
            """,
            output_dir / "country_counts.parquet",
        )
        write_title_counts(connection, output_dir)

        metadata = connection.execute(
            """
            SELECT COUNT(*)::BIGINT AS employment_spells,
                   COUNT(*) FILTER (WHERE is_us)::BIGINT AS us_employment_spells
            FROM spells
            """
        ).fetchdf()
        metadata["aggregate_schema_version"] = 1
        metadata["source_notebook"] = "E02_USPTOInventors.py"
        metadata["notebook_input_level"] = "aggregate_counts"
        metadata["available_role_columns"] = "|".join(role_columns)
        metadata["available_rics_columns"] = "|".join(rics_columns)
        metadata["classification_columns"] = "|".join(classification_columns)
        metadata["industry_columns"] = "|".join(industry_columns)
        metadata["occupation_columns"] = "|".join(occupation_columns)
        metadata["source_parquet_files"] = len(parquet_files)
        metadata.to_parquet(output_dir / "metadata.parquet", index=False)
        connection.close()


def parse_args() -> argparse.Namespace:
    """Parse the data locations and optional development row limit."""

    project_root = Path(__file__).resolve().parents[2]
    default_root = project_root / "data" / "b_temp_data" / "B02_InspectUSPTOInventors"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_root / "Inventors_USPTO_UserPositions",
    )
    parser.add_argument("--output-dir", type=Path, default=default_root)
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional development-only source-row limit; omit for production aggregates.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_aggregates(arguments.input_dir, arguments.output_dir, arguments.row_limit)
