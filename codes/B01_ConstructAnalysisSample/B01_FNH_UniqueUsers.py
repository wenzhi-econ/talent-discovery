"""
Task:
    Extract the unique users in the all-industry focal-new-hire sample.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHires_AllIndustries/*.parquet

Outputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHiresFromAllIndustries_UserID.parquet

Description of outputs:
(1) Data (a) contains one row per unique, nonmissing ``user_id`` in the input.
(2) Only ``user_id`` is read from the multi-part input dataset.

Run:
    conda run -s -n Talent python codes/B01_ConstructAnalysisSample/B01_FNH_UniqueUsers.py

Wang Wenzhi, with the help of Codex
Time: 2026-08-19
"""

import sys
from pathlib import Path
import polars as pl

CODES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES_ROOT))
import main

INPUT = main.DIR_TEMPDATA / "B01_ConstructAnalysisSample" / "FocalNewHires_AllIndustries"
OUTPUT = (
    main.DIR_TEMPDATA
    / "B01_ConstructAnalysisSample"
    / "FocalNewHiresFromAllIndustries_UserID.parquet"
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


input_files = sorted(INPUT.glob("*.parquet"))
input_data = pl.scan_parquet(INPUT / "*.parquet")

n_rows = input_data.select(pl.len()).collect(engine="streaming").item()
user_ids = input_data.select("user_id").drop_nulls().unique().sort("user_id")

user_ids.sink_parquet(
    OUTPUT,
    compression="zstd",
    engine="streaming",
)
n_unique_users = pl.scan_parquet(OUTPUT).select(pl.len()).collect(engine="streaming").item()

print(f"Read {n_rows:,} focal-new-hire rows from {len(input_files):,} parquet files.")
print(f"Saved {n_unique_users:,} unique users to {main.relative_path(OUTPUT)}.")

# Examine the generated dataset
# import pandas as pd

# df = pd.read_parquet(
#     R"E:\Dropbox\E_Projects\TalentDiscovery\data\b_temp_data\B01_ConstructAnalysisSample\FocalNewHiresFromAllIndustries_UserID.parquet"
# )
