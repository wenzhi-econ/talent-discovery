"""
Task:
    Extract the unique Revelio users linked to a PatentsView inventor.

Inputs:
(a) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/Inventors_USPTO_UserID.parquet

Description of outputs:
(1) Data (a) contains one row per unique, nonmissing ``user_id`` in the input crosswalk.
(2) The output contains only ``user_id`` and is sorted for reproducibility.

Run:
    conda run -s -n Talent python -m codes.B01_ConstructAnalysisSample.D01_GenAListOfInventors_USPTO
    conda run -s -n Talent python -m B01_ConstructAnalysisSample.D01_GenAListOfInventors_USPTO
    conda run -s -n Talent python -m D01_GenAListOfInventors_USPTO

Wang Wenzhi, with the help of Codex
Time: 2026-08-27
"""

import polars as pl

from codes import main

INPUT = main.DIR_RAWDATA / "A_Revelio" / "revelio_user_id_patentsview_id.csv"
OUTPUT = main.DIR_TEMPDATA / "B01_ConstructAnalysisSample" / "Inventors_USPTO_UserID.parquet"


if not INPUT.is_file():
    raise FileNotFoundError(f"Inventor crosswalk does not exist: {INPUT}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Polars pushes this one-column projection into the CSV scan, so the inventor-ID strings are
# not materialized when constructing the Fabric upload file.
source_user_ids = (
    pl.scan_csv(
        INPUT,
        schema_overrides={"user_id": pl.Int64},
    )
    .select("user_id")
    .collect(engine="streaming")
)

n_source_rows = source_user_ids.height
n_missing_user_ids = source_user_ids["user_id"].null_count()

user_ids = source_user_ids.drop_nulls().unique().sort("user_id")
user_ids.write_parquet(
    OUTPUT,
    compression="zstd",
    statistics=True,
)

validation = (
    pl.scan_parquet(OUTPUT)
    .select(
        pl.len().alias("rows"),
        pl.col("user_id").n_unique().alias("unique_user_ids"),
        pl.col("user_id").null_count().alias("missing_user_ids"),
    )
    .collect(engine="streaming")
    .row(0, named=True)
)

if validation["rows"] == 0:
    raise ValueError("The constructed inventor user list is empty.")
if validation["rows"] != validation["unique_user_ids"]:
    raise ValueError("The constructed inventor user list contains duplicate user IDs.")
if validation["missing_user_ids"] != 0:
    raise ValueError("The constructed inventor user list contains missing user IDs.")

print(f"Read {n_source_rows:,} inventor-link rows from {main.relative_path(INPUT)}.")
print(f"Dropped {n_missing_user_ids:,} rows with a missing user_id.")
print(f"Saved {validation['rows']:,} unique users to {main.relative_path(OUTPUT)}.")
