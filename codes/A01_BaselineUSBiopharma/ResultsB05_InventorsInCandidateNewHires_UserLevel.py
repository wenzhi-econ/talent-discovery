"""
Task:
    Summarize inventor users before and after the Stage B job-title restrictions.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsB05_InventorsInCandidateNewHires_UserLevel.csv

Descriptions of outputs:
(1) Output (a) has one row for the sample before the job-title restrictions and one row for the
    eligible sample after the restrictions.

Run:
    From codes/A01_BaselineUSBiopharma:
    conda run -s -n Talent python ResultsB05_InventorsInCandidateNewHires_UserLevel.py

Notes:
(1) The unit of each statistic is a unique user represented in the Stage B spell audit.
(2) An inventor is a user whose user_id appears in the USPTO-LinkedIn crosswalk.
(3) Users are deduplicated within each sample before counts and shares are calculated.
(4) A user is in the after-restrictions sample if at least one spell is occupation eligible.
(5) Inventor-user shares use all distinct users in the corresponding sample as denominators.


Wang Wenzhi
Time: 2026-09-22
"""

from pathlib import Path

import pandas as pd


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths and required columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "b_temp_data"
    / "A01_BaselineUSBiopharma"
    / "StageB_SpellAudit.parquet"
)
INPUT_INVENTOR_LINKS = (
    PROJECT_ROOT
    / "data"
    / "a_raw_data"
    / "A_Revelio"
    / "revelio_user_id_patentsview_id.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "A01_BaselineUSBiopharma"
    / "ResultsB05_InventorsInCandidateNewHires_UserLevel.csv"
)
AUDIT_COLUMNS = [
    "user_id",
    "id_user",
    "is_occupation_eligible",
]
INVENTOR_LINK_COLUMNS = [
    "user_id",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read and validate the inventor crosswalk
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


inventor_links = pd.read_csv(
    INPUT_INVENTOR_LINKS,
    usecols=INVENTOR_LINK_COLUMNS,
    dtype="string",
    low_memory=False,
)

if not inventor_links.columns.is_unique:
    raise ValueError("The inventor crosswalk has duplicate column names.")
inventor_links["user_id"] = inventor_links["user_id"].str.strip()
flag_missing_user_id = inventor_links["user_id"].isna() | inventor_links["user_id"].eq("")
if flag_missing_user_id.any():
    raise ValueError(
        f"The inventor crosswalk has {int(flag_missing_user_id.sum()):,} missing user IDs."
    )

flag_invalid_user_id = ~inventor_links["user_id"].str.fullmatch(r"\d+")
if flag_invalid_user_id.any():
    raise ValueError(
        f"The inventor crosswalk has {int(flag_invalid_user_id.sum()):,} nonnumeric user IDs."
    )

# A user may map to multiple PatentsView inventor IDs, but each user is counted only once.
inventor_user_ids = inventor_links["user_id"].drop_duplicates().reset_index(drop=True)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Read and validate the Stage B spell audit
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spell_audit = pd.read_parquet(INPUT_AUDIT, columns=AUDIT_COLUMNS)

if not spell_audit.columns.is_unique:
    raise ValueError("The Stage B spell audit has duplicate column names.")
if spell_audit[["user_id", "id_user"]].isna().any(axis=None):
    raise ValueError("The Stage B spell audit contains missing user identifiers.")
if spell_audit["is_occupation_eligible"].isna().any():
    raise ValueError("The Stage B eligibility indicator contains missing values.")
if not pd.api.types.is_bool_dtype(spell_audit["is_occupation_eligible"]):
    raise TypeError("The Stage B eligibility indicator must have Boolean dtype.")

source_user_ids = spell_audit["user_id"].astype("string").str.strip()
standardized_user_ids = spell_audit["id_user"].str.strip()
flag_inconsistent_user_id = source_user_ids.ne(standardized_user_ids)
if flag_inconsistent_user_id.any():
    raise ValueError(
        f"The Stage B audit has {int(flag_inconsistent_user_id.sum()):,} inconsistent user IDs."
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Calculate inventor-user counts and shares
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


sample_masks = {
    "Before job title restrictions": pd.Series(True, index=spell_audit.index),
    "After job title restrictions": spell_audit["is_occupation_eligible"],
}
summary_rows = []
for sample, sample_mask in sample_masks.items():
    sample_user_ids = standardized_user_ids.loc[sample_mask].drop_duplicates()
    is_inventor_user = sample_user_ids.isin(inventor_user_ids)
    user_count = len(sample_user_ids)
    inventor_user_count = int(is_inventor_user.sum())
    summary_rows.append(
        {
            "sample": sample,
            "user_count": user_count,
            "inventor_user_count": inventor_user_count,
            "inventor_user_share": inventor_user_count / user_count,
        }
    )

inventor_user_summary = pd.DataFrame(summary_rows)
inventor_user_summary["sample"] = inventor_user_summary["sample"].astype("string")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Save and report the summary
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
temporary_output = OUTPUT_FILE.with_suffix(f"{OUTPUT_FILE.suffix}.incomplete")
inventor_user_summary.to_csv(temporary_output, index=False)
temporary_output.replace(OUTPUT_FILE)

display_summary = inventor_user_summary.copy()
display_summary["inventor_user_share"] = display_summary["inventor_user_share"].map(
    lambda share: f"{share:.1%}"
)
print(display_summary.to_string(index=False))
print(f"Unique inventor-linked users in the crosswalk: {len(inventor_user_ids):,}")
print(f"Saved {len(inventor_user_summary):,} rows: {OUTPUT_FILE}")
