"""
Task:
    Merge the Stage C2 posting metadata and descriptions into one local Parquet file.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageC2_CandidateJobPostings/
(b) data/b_temp_data/A01_BaselineUSBiopharma/StageC2_CandidateJobPostingText/
    <== Both constructed by StageC2_CandidateUSBioPharmaJobPostings.ipynb on Fabric.

Outputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageC_CandidateLinkedInPostings.parquet

Descriptions of outputs:
(1) Output (a) has one row per eligible LinkedIn posting (`id_job`). It keeps every
    Stage C2 metadata field and appends the posting `description`.

Run:
    conda run -s -n Talent python -m codes.A01_BaselineUSBiopharma.StageC3_MergeTwoPartsTogether

Notes:
(1) Require nonmissing, unique `id_job` values in both inputs and exact key agreement.
(2) Use DuckDB to scan and merge Parquet parts without loading the full text into pandas.
(3) Sort the output by `id_job` and write one Zstandard-compressed Parquet file.
(4) Validate the staged file before atomically replacing the requested output.
(5) `OVERWRITE_OUTPUT` makes replacement behavior explicit.


Wang Wenzhi
Time: 2026-09-22
"""

import time
from pathlib import Path

import duckdb

from codes import main as project


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths and output settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


STAGE_DIRECTORY = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
INPUT_METADATA = STAGE_DIRECTORY / "StageC2_CandidateJobPostings"
INPUT_TEXT = STAGE_DIRECTORY / "StageC2_CandidateJobPostingText"
OUTPUT_FILE = STAGE_DIRECTORY / "StageC_CandidateLinkedInPostings.parquet"
STAGED_OUTPUT = OUTPUT_FILE.with_suffix(".parquet.incomplete")
OVERWRITE_OUTPUT = True
PARQUET_COMPRESSION = "zstd"
PARQUET_ROW_GROUP_SIZE = 100_000
EXPECTED_TEXT_COLUMNS = ["id_job", "description"]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Define the validated out-of-core merge
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def merge_posting_parts(
    metadata_files: list[Path],
    text_files: list[Path],
    staged_output: Path,
    *,
    compression: str,
    row_group_size: int,
) -> tuple[int, list[str]]:
    """
    Validate and merge Stage C2 metadata and text into a staged Parquet file.

    Parameters
    ----------
    metadata_files : list[Path]
        Ordered Parquet parts containing one metadata row per eligible `id_job`.
    text_files : list[Path]
        Ordered Parquet parts containing `id_job` and `description`.
    staged_output : Path
        Temporary single-file destination to validate before final replacement.
    compression : str
        Parquet compression codec; this stage permits `zstd` only.
    row_group_size : int
        Maximum number of rows in each output Parquet row group.

    Returns
    -------
    tuple[int, list[str]]
        Output row count and ordered output column names.

    Raises
    ------
    ValueError
        Raised for invalid schemas, keys, key coverage, or settings.
    AssertionError
        Raised when the staged output differs from the validated merge contract.

    Notes
    -----
    (1) DuckDB streams the Parquet inputs and avoids materializing the full text in pandas.
    (2) Sorting by `id_job` makes the single-file output order deterministic.
    """
    if compression != "zstd":
        raise ValueError("This stage supports only Zstandard Parquet compression.")
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive.")

    connection = duckdb.connect()
    try:
        metadata = connection.read_parquet([str(path) for path in metadata_files])
        posting_text = connection.read_parquet([str(path) for path in text_files])
        metadata_columns = list(metadata.columns)
        metadata_types = list(metadata.types)
        text_columns = list(posting_text.columns)
        text_types = list(posting_text.types)

        if len(metadata_columns) != len(set(metadata_columns)):
            raise ValueError("Stage C2 metadata contains duplicate column names.")
        if "id_job" not in metadata_columns or "description" in metadata_columns:
            raise ValueError("Metadata must contain id_job and exclude description.")
        if text_columns != EXPECTED_TEXT_COLUMNS:
            raise ValueError(
                "Posting text columns must be exactly "
                f"{EXPECTED_TEXT_COLUMNS}; found {text_columns}."
            )

        metadata.create_view("candidate_metadata", replace=True)
        posting_text.create_view("candidate_text", replace=True)
        metadata_count, metadata_nonmissing, metadata_unique = connection.execute(
            """
            SELECT COUNT(*), COUNT(id_job), COUNT(DISTINCT id_job)
            FROM candidate_metadata
            """
        ).fetchone()
        text_count, text_nonmissing, text_unique = connection.execute(
            """
            SELECT COUNT(*), COUNT(id_job), COUNT(DISTINCT id_job)
            FROM candidate_text
            """
        ).fetchone()
        if metadata_count == 0 or metadata_count != metadata_nonmissing:
            raise ValueError("Stage C2 metadata is empty or has missing id_job values.")
        if text_count == 0 or text_count != text_nonmissing:
            raise ValueError("Stage C2 text is empty or has missing id_job values.")
        if metadata_count != metadata_unique or text_count != text_unique:
            raise ValueError("Each Stage C2 input must have one row per unique id_job.")

        missing_metadata, missing_text = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE metadata.id_job IS NULL),
                COUNT(*) FILTER (WHERE posting_text.id_job IS NULL)
            FROM candidate_metadata AS metadata
            FULL OUTER JOIN candidate_text AS posting_text USING (id_job)
            """
        ).fetchone()
        if missing_metadata or missing_text:
            raise ValueError(
                "Stage C2 keys disagree: "
                f"{missing_metadata:,} lack metadata and {missing_text:,} lack text."
            )

        output_columns = [*metadata_columns, "description"]
        staged_sql_path = staged_output.as_posix().replace("'", "''")
        connection.execute(
            f"""
            COPY (
                SELECT metadata.*, posting_text.description
                FROM candidate_metadata AS metadata
                INNER JOIN candidate_text AS posting_text USING (id_job)
                ORDER BY metadata.id_job
            ) TO '{staged_sql_path}' (
                FORMAT PARQUET,
                COMPRESSION {compression},
                ROW_GROUP_SIZE {row_group_size}
            )
            """
        )

        written = connection.read_parquet(str(staged_output))
        if list(written.columns) != output_columns:
            raise AssertionError("Staged output columns differ from the merge contract.")
        if list(written.types) != [*metadata_types, text_types[1]]:
            raise AssertionError("Staged output types differ from the input types.")
        written.create_view("written_postings", replace=True)
        written_count, written_nonmissing, written_unique = connection.execute(
            """
            SELECT COUNT(*), COUNT(id_job), COUNT(DISTINCT id_job)
            FROM written_postings
            """
        ).fetchone()
        if (
            written_count != metadata_count
            or written_nonmissing != written_count
            or written_unique != written_count
        ):
            raise AssertionError("Staged output row count or id_job key is invalid.")
        return written_count, output_columns
    finally:
        connection.close()


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Validate inputs and write the merged posting file
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
project.ensure_parent(OUTPUT_FILE)
if OUTPUT_FILE.exists() and not OVERWRITE_OUTPUT:
    raise FileExistsError(OUTPUT_FILE)
if not INPUT_METADATA.is_dir() or not INPUT_TEXT.is_dir():
    raise FileNotFoundError(
        f"Expected Stage C2 input directories: {INPUT_METADATA} and {INPUT_TEXT}."
    )

metadata_files = sorted(INPUT_METADATA.glob("*.parquet"))
text_files = sorted(INPUT_TEXT.glob("*.parquet"))
if not metadata_files or not text_files:
    raise FileNotFoundError("Each Stage C2 input directory must contain Parquet parts.")
if STAGED_OUTPUT.exists():
    STAGED_OUTPUT.unlink()

posting_count, output_columns = merge_posting_parts(
    metadata_files,
    text_files,
    STAGED_OUTPUT,
    compression=PARQUET_COMPRESSION,
    row_group_size=PARQUET_ROW_GROUP_SIZE,
)

# Preserve the previous valid output until the staged replacement has passed every check.
for attempt in range(5):
    try:
        STAGED_OUTPUT.replace(OUTPUT_FILE)
        break
    except PermissionError:
        if attempt == 4:
            raise
        time.sleep(0.25 * 2**attempt)

output_size_mb = OUTPUT_FILE.stat().st_size / 1024**2
project.finish_run(
    run,
    f"Saved {posting_count:,} posting rows and {len(output_columns)} columns.",
    f"Output size: {output_size_mb:,.1f} MiB.",
    f"Output: {project.relative_path(OUTPUT_FILE)}",
)
