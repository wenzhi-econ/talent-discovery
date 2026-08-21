"""
Task:
    Generate a unique ID for each distinct job description.

Inputs:
(a) data/b_temp_data/C01_PreProcessPostings/TestSample.parquet

Outputs:
(a) data/b_temp_data/C01_PreProcessPostings/A_HashMapFromPostingsToDescriptions.parquet
(b) data/b_temp_data/C01_PreProcessPostings/A_UniqueDescriptions.parquet

Description of the outputs:
(1) Output (a) is at posting level, i.e., each row is one job posting as in the input.
(2) Output (b) is at description level, i.e., each row is one unique job description.

Run:
    conda run -s -n Talent python codes/C01_PreProcessPostings/A_HashDescriptions.py

Wang Wenzhi, with the help of Codex
Time: 2026-08-18
"""

import hashlib
from pathlib import Path
from decimal import Decimal
from typing import Final, TypeAlias, TypedDict, cast
import sys
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _Utilities_C01 as util


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify global parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-1. General settings
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


BATCH_SIZE = 20_000
COMPRESSION = "zstd"
IF_OVERWRITE = True
IF_DELETETABLE = True


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-2. Input and output datasets
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

FOLDER_DATA: Final[Path] = util.FOLDER_DATA
INPUT: Final[Path] = FOLDER_DATA / "TestSample.parquet"
OUT_MAP: Final[Path] = FOLDER_DATA / "A_HashMapFromPostingsToDescriptions.parquet"
OUT_DESCRIPTIONS: Final[Path] = FOLDER_DATA / "A_UniqueDescriptions.parquet"
TEMP_TABLE: Final[Path] = FOLDER_DATA / "A_Table.db"

LABEL_MAP = {"job_id": "ID for the job postings", "description_hash": "ID for job descriptions"}
SCHEMA_MAP: Final[pa.Schema] = pa.schema(
    [
        ("job_id", pa.string(), False),
        ("description_hash", pa.string()),
    ]
)

LABEL_DESCRIPTIONS = {
    "description_hash": "ID for job descriptions",
    "description": "Raw job descriptions",
    "description_multiplicity": "The number of postings with the same raw job description",
    "first_job_id": "The first `job_id` with the same raw job description",
}
SCHEMA_DESCRIPTIONS: Final[pa.Schema] = pa.schema(
    [
        ("description_hash", pa.string(), False),
        ("description", pa.large_string(), False),
        ("description_multiplicity", pa.int64(), False),
        ("first_job_id", pa.string(), False),
    ]
)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-3. Useful settings for type hinting
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


T_JobId: TypeAlias = int | Decimal | None


class T_Postings(TypedDict):
    """Fields read from one input posting."""

    job_id: T_JobId
    description: str | None


class T_Maps(TypedDict):
    """Fields written to the posting-to-description map."""

    job_id: str
    description_hash: str | None


class T_Descriptions(TypedDict):
    """Fields written for one unique description."""

    description_hash: str
    description: str
    description_multiplicity: int
    first_job_id: str


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Prepare for data transformation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


time_start = util.record_time()
util.report_status(text=f"Start running {Path(__file__).name} at {time_start}")

temp_map: Path = util.prepare_output(outpath=OUT_MAP, overwrite=IF_OVERWRITE)
temp_descriptions: Path = util.prepare_output(outpath=OUT_DESCRIPTIONS, overwrite=IF_OVERWRITE)

writer_map = pq.ParquetWriter(temp_map, SCHEMA_MAP, compression=COMPRESSION)
writer_descriptions = pq.ParquetWriter(
    temp_descriptions, SCHEMA_DESCRIPTIONS, compression=COMPRESSION
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read postings in batch and record unique descriptions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Initialize the temporary table
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_create_descriptions_table = R"""
CREATE TABLE tab_descriptions (
    description_hash TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    description_multiplicity INTEGER NOT NULL,
    first_job_id TEXT NOT NULL
);
"""

con = util.create_sqlite_table(
    tabpath=TEMP_TABLE,
    sql_command=sql_create_descriptions_table,
)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Define helper functions for data transformation
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def make_string_job_id(value: T_JobId) -> str:
    """
    Transform ``job_id`` as a string.
    """
    if value is None:
        raise ValueError("job_id is null")
    if isinstance(value, Decimal):
        if value != value.to_integral_value():
            raise ValueError(f"Non-integral job_id: {value}")
        return format(value, "f")
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"Unexpected job_id type: {type(value).__name__}")


def generate_description_hash(value: str) -> str:
    """
    Generate a hash ID for description texts.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-3. Read the input parquet file in batch and write to ``OUT_MAP``
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_select_description_by_hash = R"""
SELECT description
FROM tab_descriptions
WHERE description_hash = ?;
"""

sql_insert_description = R"""
INSERT INTO tab_descriptions (
    description_hash,
    description,
    description_multiplicity,
    first_job_id
)
VALUES (?, ?, 1, ?);
"""

sql_increment_multiplicity = R"""
UPDATE tab_descriptions
SET description_multiplicity = description_multiplicity + 1
WHERE description_hash = ?;
"""

input_file = pq.ParquetFile(INPUT)

n_postings = 0
n_notnull_postings = 0
for batch in input_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["job_id", "description"],
):
    out_map_rows: list[T_Maps] = []
    for record in cast(list[T_Postings], batch.to_pylist()):
        # Generate description hash.
        _job_id = record["job_id"]
        _description = record["description"]
        job_id = make_string_job_id(_job_id)
        description_hash = (
            generate_description_hash(_description) if isinstance(_description, str) else None
        )
        out_map_rows.append({"job_id": job_id, "description_hash": description_hash})
        # Check if the description shows up before for only non-NULL ``_description``
        if description_hash is not None:
            n_notnull_postings += 1
            existing: tuple[str] | None = con.execute(
                sql_select_description_by_hash,
                (description_hash,),
            ).fetchone()
            # If the ``description_hash`` never shows up before, add it to ``tab_descriptions``.
            # Otherwise, add 1 to the ``description_multiplicity`` column.
            if existing is None:
                con.execute(
                    sql_insert_description,
                    (description_hash, _description, job_id),
                )
            else:
                if existing[0] != _description:
                    raise RuntimeError("SHA-256 collision detected")
                con.execute(sql_increment_multiplicity, (description_hash,))
    writer_map.write_table(pa.Table.from_pylist(out_map_rows, schema=SCHEMA_MAP))
    n_postings += len(out_map_rows)
    con.commit()
    print(f"Processed posting rows: {n_postings:,}")

writer_map.close()
print(f"\nFinished writing to the temp dataset: {util.relative_path(temp_map)}.")


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-4. Write unique descriptions to ``OUT_DESCRIPTIONS``
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_select_descriptions = R"""
SELECT
    description_hash,
    description,
    description_multiplicity,
    first_job_id
FROM tab_descriptions
ORDER BY description_hash;
"""

cursor = con.execute(sql_select_descriptions)

n_descriptions: int = 0

while rows := cursor.fetchmany(BATCH_SIZE):
    list_descriptions: list[T_Descriptions] = [
        {
            "description_hash": row[0],
            "description": row[1],
            "description_multiplicity": row[2],
            "first_job_id": row[3],
        }
        for row in rows
    ]
    writer_descriptions.write_table(
        pa.Table.from_pylist(list_descriptions, schema=SCHEMA_DESCRIPTIONS)
    )
    n_descriptions += len(rows)

writer_descriptions.close()
print(f"\nFinished writing to the temp dataset: {util.relative_path(temp_descriptions)}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Do simple diagnostics and save the final datasets
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-1. Do simple diagnostics
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_sum_multiplicity = R"""
SELECT COALESCE(SUM(description_multiplicity), 0)
FROM tab_descriptions;
"""

multiplicity = con.execute(sql_sum_multiplicity).fetchone()[0]
con.close()

if IF_DELETETABLE:
    # The database file will be removed from the disk if ``IF_DELETETABLE``.
    TEMP_TABLE.unlink()

assert n_postings == input_file.metadata.num_rows
assert pq.ParquetFile(temp_map).metadata.num_rows == n_postings
assert pq.ParquetFile(temp_descriptions).metadata.num_rows == n_descriptions
assert multiplicity == n_notnull_postings


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-2. Save the final datasets
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


util.publish_output(temp_output=temp_map, final_output=OUT_MAP)
print(f"\nSaved the final dataset: {util.relative_path(OUT_MAP)}.")
print(f"Variables in the dataset: {LABEL_MAP}.")

util.publish_output(temp_output=temp_descriptions, final_output=OUT_DESCRIPTIONS)
print(f"\nSaved the final dataset: {util.relative_path(OUT_DESCRIPTIONS)}.")
print(f"Variables in the dataset: {LABEL_DESCRIPTIONS}.")

print(f"\nIn total, there are {n_postings:,} postings and {n_descriptions:,} descriptions.")


time_end = util.record_time()
util.report_status(
    text=(
        f"Finished running {Path(__file__).name} at {time_start}\n"
        f"Time used: {time_end - time_start}"
    )
)
