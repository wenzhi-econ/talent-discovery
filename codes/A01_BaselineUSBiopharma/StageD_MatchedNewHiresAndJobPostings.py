"""
Task:
    Match candidate new hires to corresponding prior job postings within each company.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_CandidateNewHires.parquet
    <== Constructed by StageB_CandidateUSBiopharmaNewHires.py.
(b) data/b_temp_data/A01_BaselineUSBiopharma/StageC_CandidateLinkedInPostings.parquet
    <== Constructed by StageC3_MergeTwoPartsTogether.py.

Outputs:
(a) data/c_final_data/A01_BaselineUSBiopharma/StageD_HirePostingLinks.parquet
(b) data/c_final_data/A01_BaselineUSBiopharma/StageD_MatchedNewHires.parquet
(c) data/c_final_data/A01_BaselineUSBiopharma/StageD_MatchedJobPostings.parquet
(d) data/c_final_data/A01_BaselineUSBiopharma/StageD_FullCandidateNewHires.parquet
(e) data/c_final_data/A01_BaselineUSBiopharma/StageD_FullCandidateJobPostings.parquet

Descriptions of outputs:
(1) Output (a) has one row per unique eligible hire-posting pair, with link diagnostics.
(2) Outputs (b) and (d) have one row per user-company, for matched and full samples.
(3) Outputs (c) and (e) have one row per posting, for matched and full samples.
(4) Both full samples retain all source fields, partner counts, and match status.

Run:
    conda run -s -n Talent python -m codes.A01_BaselineUSBiopharma.StageD_MatchedNewHiresAndJobPostings

Notes:
(1) Link within company when h minus 12 months <= p <= h minus 1 month, inclusive.
(2) Calendar subtraction clips to the last valid target day. No state, title, or seniority
    restriction is imposed.
(3) State, title, and seniority equality is nullable when either field is unavailable.
(4) Posting seniority is absent in the documented source; do not infer it from job_category.
(5) Links describe recruitment exposure, not verified vacancies or hiring assignments.
(6) Hire dates are assumed daily; posting precision and month-date treatment are retained.
(7) DuckDB performs the range join, aggregation, and direct Parquet writes out of core.
(8) Link row order is not part of the output contract; endpoint outputs are sorted by key.
(9) Outputs are staged and validated before publication. A stale incomplete file blocks a run.
(10) Existing final outputs require OVERWRITE_OUTPUTS = True.
(11) Full career histories, coverage, title validity, and follow-up remain unvalidated.


Wang Wenzhi
Time: 2026-09-22
"""

import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from codes import main as project


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths, settings, and stage-specific helper functions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_DIRECTORY = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
INPUT_HIRES = INPUT_DIRECTORY / "StageB_CandidateNewHires.parquet"
INPUT_POSTINGS = INPUT_DIRECTORY / "StageC_CandidateLinkedInPostings.parquet"
OUTPUT_DIRECTORY = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
OUTPUT_LINKS = OUTPUT_DIRECTORY / "StageD_HirePostingLinks.parquet"
OUTPUT_MATCHED_HIRES = OUTPUT_DIRECTORY / "StageD_MatchedNewHires.parquet"
OUTPUT_MATCHED_POSTINGS = OUTPUT_DIRECTORY / "StageD_MatchedJobPostings.parquet"
OUTPUT_FULL_HIRES = OUTPUT_DIRECTORY / "StageD_FullCandidateNewHires.parquet"
OUTPUT_FULL_POSTINGS = OUTPUT_DIRECTORY / "StageD_FullCandidateJobPostings.parquet"
OUTPUT_FILES = [
    OUTPUT_LINKS,
    OUTPUT_MATCHED_HIRES,
    OUTPUT_MATCHED_POSTINGS,
    OUTPUT_FULL_HIRES,
    OUTPUT_FULL_POSTINGS,
]
STAGED_OUTPUTS = {
    output_file: output_file.with_suffix(".parquet.incomplete") for output_file in OUTPUT_FILES
}
OVERWRITE_OUTPUTS = False
PARQUET_COMPRESSION = "zstd"
PARQUET_ROW_GROUP_SIZE = 500_000
LINK_RULE_VERSION = "company_calendar_1_12_inclusive_v1"
MISSING_TEXT = ("", "empty", "null", "none", "nan", "na", "n/a")
LINK_COLUMNS = [
    "id_user",
    "id_rcid",
    "id_position",
    "id_job",
    "start_date",
    "publication_date",
    "posting_lower",
    "posting_upper",
    "lag_days",
    "lag_calendar_months",
    "is_state_equal",
    "is_title_equal",
    "is_seniority_equal",
    "hire_date_precision",
    "posting_date_precision",
    "link_rule_version",
]


def sql_string(value: str | Path) -> str:
    """
    Quote a path or string as a DuckDB SQL literal.

    Parameters
    ----------
    value : str or pathlib.Path
        Value inserted into a generated SQL statement.

    Returns
    -------
    str
        Single-quoted SQL literal with embedded quotes escaped.
    """
    text = str(value).replace("\\", "/").replace("'", "''")
    return f"'{text}'"


def parquet_columns(
    connection: duckdb.DuckDBPyConnection,
    parquet_file: Path,
) -> list[str]:
    """
    Return ordered column names from a Parquet dataset.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    parquet_file : pathlib.Path
        Single Parquet file whose schema is inspected.

    Returns
    -------
    list[str]
        Ordered field names reported by DuckDB.
    """
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(parquet_file)})"
    ).fetchall()
    return [row[0] for row in rows]


def require_columns(
    observed_columns: list[str],
    required_columns: list[str],
    *,
    source: Path,
) -> None:
    """
    Require named fields and reject duplicate column labels.

    Parameters
    ----------
    observed_columns : list[str]
        Ordered fields found in the input.
    required_columns : list[str]
        Fields required by the matching workflow.
    source : pathlib.Path
        Input path included in any error message.
    """
    missing = sorted(set(required_columns) - set(observed_columns))
    if missing or len(observed_columns) != len(set(observed_columns)):
        raise ValueError(
            f"Invalid schema in {source}: missing columns {missing}; "
            "column names must also be unique."
        )


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> int:
    """
    Execute a scalar count query and return an integer.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    query : str
        SQL query returning one nonmissing scalar value.

    Returns
    -------
    int
        Scalar query result converted to a Python integer.
    """
    value = connection.execute(query).fetchone()[0]
    if value is None:
        raise ValueError("A required scalar query returned NULL.")
    return int(value)


def require_zero(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    message: str,
) -> None:
    """
    Raise when a diagnostic query finds one or more invalid rows.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    query : str
        SQL count query identifying invalid observations.
    message : str
        Explanation included with the invalid-row count.
    """
    invalid_count = scalar(connection, query)
    if invalid_count:
        raise ValueError(f"{message} Invalid rows: {invalid_count:,}.")


def normalized_text_expression(column: str | None) -> str:
    """
    Construct nullable lowercase text for a DuckDB diagnostic comparison.

    Parameters
    ----------
    column : str or None
        Qualified SQL field name, or None when the source field is unavailable.

    Returns
    -------
    str
        SQL expression that trims text and converts explicit missing tokens to NULL.
    """
    if column is None:
        return "CAST(NULL AS VARCHAR)"
    normalized = f"lower(trim(CAST({column} AS VARCHAR)))"
    missing_values = ", ".join(sql_string(value) for value in MISSING_TEXT)
    return f"CASE WHEN {normalized} IN ({missing_values}) THEN NULL ELSE {normalized} END"


def build_link_query(
    hire_columns: list[str],
    posting_columns: list[str],
) -> str:
    """
    Build the DuckDB range-join query implementing the Stage D matching rule.

    Parameters
    ----------
    hire_columns : list[str]
        Fields available in the Stage B candidate-hire input.
    posting_columns : list[str]
        Fields available in the Stage C candidate-posting input.

    Returns
    -------
    str
        SQL query producing the complete Stage D link schema.

    Notes
    -----
    (1) State, title, and seniority comparisons are diagnostics and never filter links.
    (2) SQL equality returns NULL when either normalized diagnostic value is NULL.
    """
    hire_state = normalized_text_expression("h.state" if "state" in hire_columns else None)
    posting_state = normalized_text_expression("p.state" if "state" in posting_columns else None)
    hire_title = normalized_text_expression(
        "h.job_title_normalized" if "job_title_normalized" in hire_columns else None
    )
    posting_title = normalized_text_expression(
        "p.job_title_normalized" if "job_title_normalized" in posting_columns else None
    )
    hire_seniority = normalized_text_expression(
        "h.seniority" if "seniority" in hire_columns else None
    )
    posting_seniority = normalized_text_expression(
        "p.seniority" if "seniority" in posting_columns else None
    )
    return f"""
        WITH prepared_hires AS (
            SELECT
                CAST(id_user AS VARCHAR) AS id_user,
                CAST(id_rcid AS VARCHAR) AS id_rcid,
                CAST(id_position AS VARCHAR) AS id_position,
                CAST(start_date AS TIMESTAMP_NS) AS start_date,
                CAST(start_date - INTERVAL 12 MONTH AS TIMESTAMP_NS) AS posting_lower,
                CAST(start_date - INTERVAL 1 MONTH AS TIMESTAMP_NS) AS posting_upper,
                {hire_state} AS diagnostic_state,
                {hire_title} AS diagnostic_title,
                {hire_seniority} AS diagnostic_seniority
            FROM candidate_hires AS h
        ),
        prepared_postings AS (
            SELECT
                CAST(id_rcid AS VARCHAR) AS id_rcid,
                CAST(id_job AS VARCHAR) AS id_job,
                CAST(publication_date AS TIMESTAMP_NS) AS publication_date,
                CAST(publication_date_precision AS VARCHAR) AS publication_date_precision,
                {posting_state} AS diagnostic_state,
                {posting_title} AS diagnostic_title,
                {posting_seniority} AS diagnostic_seniority
            FROM candidate_postings AS p
        )
        SELECT
            h.id_user,
            h.id_rcid,
            h.id_position,
            p.id_job,
            h.start_date,
            p.publication_date,
            h.posting_lower,
            h.posting_upper,
            CAST(date_diff('day', p.publication_date, h.start_date) AS BIGINT) AS lag_days,
            CAST(
                (year(h.start_date) - year(p.publication_date)) * 12
                + month(h.start_date) - month(p.publication_date)
                AS BIGINT
            ) AS lag_calendar_months,
            p.diagnostic_state = h.diagnostic_state AS is_state_equal,
            p.diagnostic_title = h.diagnostic_title AS is_title_equal,
            p.diagnostic_seniority = h.diagnostic_seniority AS is_seniority_equal,
            CAST('day_representation' AS VARCHAR) AS hire_date_precision,
            p.publication_date_precision AS posting_date_precision,
            CAST({sql_string(LINK_RULE_VERSION)} AS VARCHAR) AS link_rule_version
        FROM prepared_hires AS h
        INNER JOIN prepared_postings AS p
            ON h.id_rcid = p.id_rcid
            AND p.publication_date BETWEEN h.posting_lower AND h.posting_upper
    """


def copy_query_to_parquet(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_file: Path,
) -> None:
    """
    Execute a query and stream its result directly to one staged Parquet file.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    query : str
        Query whose full result is written without conversion to pandas.
    output_file : pathlib.Path
        Staged Parquet destination, which must not already exist.
    """
    if output_file.exists():
        raise FileExistsError(f"Staged output already exists: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"""
        COPY ({query})
        TO {sql_string(output_file)} (
            FORMAT PARQUET,
            COMPRESSION {PARQUET_COMPRESSION},
            ROW_GROUP_SIZE {PARQUET_ROW_GROUP_SIZE}
        )
        """
    )


def parquet_row_count(parquet_file: Path) -> int:
    """
    Return the row count stored in Parquet metadata.

    Parameters
    ----------
    parquet_file : pathlib.Path
        Closed and readable Parquet file.

    Returns
    -------
    int
        Number of rows across all row groups.
    """
    return pq.ParquetFile(parquet_file).metadata.num_rows


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Guard outputs and validate both candidate inputs with DuckDB
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
for output_file in OUTPUT_FILES:
    if output_file.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(output_file)
for staged_output in STAGED_OUTPUTS.values():
    if staged_output.exists():
        raise FileExistsError(
            f"Incomplete Stage D output exists; confirm no run is active, then remove it: "
            f"{staged_output}"
        )

connection = duckdb.connect()
connection.execute("SET enable_progress_bar = true")
connection.execute("SET enable_progress_bar_print = true")
connection.execute("SET preserve_insertion_order = false")
connection.execute(
    f"""
    CREATE TEMP VIEW candidate_hires AS
    SELECT * FROM read_parquet({sql_string(INPUT_HIRES)})
    """
)
connection.execute(
    f"""
    CREATE TEMP VIEW candidate_postings AS
    SELECT * FROM read_parquet({sql_string(INPUT_POSTINGS)})
    """
)

hire_columns = parquet_columns(connection, INPUT_HIRES)
posting_columns = parquet_columns(connection, INPUT_POSTINGS)
require_columns(
    hire_columns,
    [
        "id_user",
        "id_rcid",
        "id_position",
        "start_date",
        "is_occupation_eligible",
        "country",
    ],
    source=INPUT_HIRES,
)
require_columns(
    posting_columns,
    [
        "id_rcid",
        "id_job",
        "publication_date",
        "publication_date_precision",
        "month_only_policy",
        "is_occupation_eligible",
        "country",
    ],
    source=INPUT_POSTINGS,
)
for reserved_column, source_columns in [
    ("posting_partner_count", hire_columns),
    ("is_matched", hire_columns),
    ("hire_partner_count", posting_columns),
    ("is_matched", posting_columns),
]:
    if reserved_column in source_columns:
        raise ValueError(f"Constructed output column already exists: {reserved_column}")

hire_count = scalar(connection, "SELECT count(*) FROM candidate_hires")
posting_count = scalar(connection, "SELECT count(*) FROM candidate_postings")
if hire_count == 0 or posting_count == 0:
    raise ValueError("Candidate hire and posting inputs must both be nonempty.")
for view_name, key in [
    ("candidate_hires", "id_position"),
    ("candidate_postings", "id_job"),
]:
    key_count = scalar(
        connection,
        f"SELECT count(DISTINCT {key}) FROM {view_name} WHERE {key} IS NOT NULL",
    )
    if key_count != scalar(connection, f"SELECT count({key}) FROM {view_name}"):
        raise ValueError(f"Missing or duplicate key in {view_name}: {key}.")
pair_count = scalar(
    connection,
    """
    SELECT count(DISTINCT (id_user, id_rcid))
    FROM candidate_hires
    WHERE id_user IS NOT NULL AND id_rcid IS NOT NULL
    """,
)
if pair_count != hire_count:
    raise ValueError("Stage B must contain one row per nonmissing user-company pair.")

identifier_checks = [
    ("candidate_hires", "id_user"),
    ("candidate_hires", "id_rcid"),
    ("candidate_hires", "id_position"),
    ("candidate_postings", "id_rcid"),
    ("candidate_postings", "id_job"),
]
for view_name, column in identifier_checks:
    require_zero(
        connection,
        f"""
        SELECT count(*)
        FROM {view_name}
        WHERE {column} IS NULL
            OR NOT regexp_full_match(
                CAST({column} AS VARCHAR),
                '-?(0|[1-9][0-9]*)'
            )
        """,
        f"{view_name}.{column} must contain canonical signed integer strings.",
    )
for view_name in ["candidate_hires", "candidate_postings"]:
    require_zero(
        connection,
        f"""
        SELECT count(*)
        FROM {view_name}
        WHERE is_occupation_eligible IS DISTINCT FROM TRUE
        """,
        f"Occupation-ineligible rows reached {view_name}.",
    )
    require_zero(
        connection,
        f"""
        SELECT count(*)
        FROM {view_name}
        WHERE country IS NULL OR trim(CAST(country AS VARCHAR)) <> 'United States'
        """,
        f"Non-US rows reached {view_name}.",
    )
require_zero(
    connection,
    """
    SELECT count(*)
    FROM candidate_hires
    WHERE start_date IS NULL OR start_date <> date_trunc('day', start_date)
    """,
    "Hire dates must be nonmissing daily dates.",
)
require_zero(
    connection,
    "SELECT count(*) FROM candidate_postings WHERE publication_date IS NULL",
    "Posting dates must be nonmissing.",
)
date_policies = {
    row[0]
    for row in connection.execute(
        "SELECT DISTINCT month_only_policy FROM candidate_postings"
    ).fetchall()
}
if len(date_policies) != 1 or not date_policies.issubset({"error", "first_day"}):
    raise ValueError(f"Unexpected posting month-only date policies: {date_policies}")
print(f"Validated {hire_count:,} candidate hires and {posting_count:,} candidate postings.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Construct and stream the many-to-many link table with DuckDB
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


link_query = build_link_query(hire_columns, posting_columns)
connection.execute(f"CREATE TEMP VIEW eligible_links AS {link_query}")
link_count = scalar(connection, "SELECT count(*) FROM eligible_links")
if link_count == 0:
    raise ValueError("The matching rule generated no hire-posting links.")
print(f"The matching rule generates {link_count:,} links. Writing the staged link file.")
step_started = time.perf_counter()
copy_query_to_parquet(
    connection,
    "SELECT * FROM eligible_links",
    STAGED_OUTPUTS[OUTPUT_LINKS],
)
if parquet_row_count(STAGED_OUTPUTS[OUTPUT_LINKS]) != link_count:
    raise AssertionError("The staged link row count differs from the range-join count.")
if parquet_columns(connection, STAGED_OUTPUTS[OUTPUT_LINKS]) != LINK_COLUMNS:
    raise AssertionError("The staged link columns differ from the declared output contract.")
print(f"Wrote and validated the link table in {time.perf_counter() - step_started:,.1f} seconds.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Derive partner counts and stage full and matched endpoint samples
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


step_started = time.perf_counter()
connection.execute(
    f"""
    CREATE TEMP TABLE endpoint_counts AS
    SELECT
        id_position,
        id_job,
        CAST(count(*) AS BIGINT) AS partner_count
    FROM read_parquet({sql_string(STAGED_OUTPUTS[OUTPUT_LINKS])})
    GROUP BY GROUPING SETS ((id_position), (id_job))
    """
)
connection.execute(
    """
    CREATE TEMP VIEW hire_counts AS
    SELECT id_position, partner_count AS posting_partner_count
    FROM endpoint_counts
    WHERE id_position IS NOT NULL
    """
)
connection.execute(
    """
    CREATE TEMP VIEW posting_counts AS
    SELECT id_job, partner_count AS hire_partner_count
    FROM endpoint_counts
    WHERE id_job IS NOT NULL
    """
)
hire_link_sum = scalar(connection, "SELECT sum(posting_partner_count) FROM hire_counts")
posting_link_sum = scalar(connection, "SELECT sum(hire_partner_count) FROM posting_counts")
if hire_link_sum != link_count or posting_link_sum != link_count:
    raise AssertionError("Endpoint partner counts do not sum to the link-table row count.")
matched_hire_count = scalar(connection, "SELECT count(*) FROM hire_counts")
matched_posting_count = scalar(connection, "SELECT count(*) FROM posting_counts")

full_hire_query = """
    SELECT
        hires.*,
        CAST(coalesce(counts.posting_partner_count, 0) AS BIGINT) AS posting_partner_count,
        coalesce(counts.posting_partner_count, 0) > 0 AS is_matched
    FROM candidate_hires AS hires
    LEFT JOIN hire_counts AS counts USING (id_position)
    ORDER BY hires.id_user, hires.id_rcid
"""
full_posting_query = """
    SELECT
        postings.*,
        CAST(coalesce(counts.hire_partner_count, 0) AS BIGINT) AS hire_partner_count,
        coalesce(counts.hire_partner_count, 0) > 0 AS is_matched
    FROM candidate_postings AS postings
    LEFT JOIN posting_counts AS counts USING (id_job)
    ORDER BY postings.id_rcid, postings.id_job
"""
copy_query_to_parquet(
    connection,
    full_hire_query,
    STAGED_OUTPUTS[OUTPUT_FULL_HIRES],
)
copy_query_to_parquet(
    connection,
    full_posting_query,
    STAGED_OUTPUTS[OUTPUT_FULL_POSTINGS],
)
copy_query_to_parquet(
    connection,
    f"""
    SELECT *
    FROM read_parquet({sql_string(STAGED_OUTPUTS[OUTPUT_FULL_HIRES])})
    WHERE is_matched
    ORDER BY id_user, id_rcid
    """,
    STAGED_OUTPUTS[OUTPUT_MATCHED_HIRES],
)
copy_query_to_parquet(
    connection,
    f"""
    SELECT *
    FROM read_parquet({sql_string(STAGED_OUTPUTS[OUTPUT_FULL_POSTINGS])})
    WHERE is_matched
    ORDER BY id_rcid, id_job
    """,
    STAGED_OUTPUTS[OUTPUT_MATCHED_POSTINGS],
)

expected_rows = {
    OUTPUT_FULL_HIRES: hire_count,
    OUTPUT_FULL_POSTINGS: posting_count,
    OUTPUT_MATCHED_HIRES: matched_hire_count,
    OUTPUT_MATCHED_POSTINGS: matched_posting_count,
}
expected_columns = {
    OUTPUT_FULL_HIRES: hire_columns + ["posting_partner_count", "is_matched"],
    OUTPUT_MATCHED_HIRES: hire_columns + ["posting_partner_count", "is_matched"],
    OUTPUT_FULL_POSTINGS: posting_columns + ["hire_partner_count", "is_matched"],
    OUTPUT_MATCHED_POSTINGS: posting_columns + ["hire_partner_count", "is_matched"],
}
for output_file, expected_count in expected_rows.items():
    staged_output = STAGED_OUTPUTS[output_file]
    if parquet_row_count(staged_output) != expected_count:
        raise AssertionError(f"Unexpected staged row count: {staged_output}")
    if parquet_columns(connection, staged_output) != expected_columns[output_file]:
        raise AssertionError(f"Unexpected staged schema: {staged_output}")
for staged_output, count_column in [
    (STAGED_OUTPUTS[OUTPUT_FULL_HIRES], "posting_partner_count"),
    (STAGED_OUTPUTS[OUTPUT_FULL_POSTINGS], "hire_partner_count"),
]:
    require_zero(
        connection,
        f"""
        SELECT count(*)
        FROM read_parquet({sql_string(staged_output)})
        WHERE {count_column} IS NULL
            OR {count_column} < 0
            OR is_matched IS NULL
            OR is_matched <> ({count_column} > 0)
        """,
        f"Partner counts and match status disagree in {staged_output}.",
    )
print(
    "Constructed and validated endpoint outputs in "
    f"{time.perf_counter() - step_started:,.1f} seconds."
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Publish every validated output and report completion
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


connection.close()
for output_file in OUTPUT_FILES:
    STAGED_OUTPUTS[output_file].replace(output_file)
    print(f"Published {parquet_row_count(output_file):,} rows: {output_file}")
project.finish_run(
    run,
    f"Saved {link_count:,} unique eligible pairs: {OUTPUT_LINKS}.",
    f"Matched {matched_hire_count:,} hires and {matched_posting_count:,} postings.",
)
