"""
Task:
    Summarize inventor employment spells before and after the Stage B job-title restrictions.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsB04_InventorsInCandidateNewHires.csv

Descriptions of outputs:
(1) Output (a) has one row for the sample before the job-title restrictions and one row for the
    eligible sample after the restrictions.

Run:
    From codes/A01_BaselineUSBiopharma:
    conda run -s -n Talent python ResultsB04_InventorsInCandidateNewHires.py

Notes:
(1) The unit is an employment spell in the Stage B spell audit.
(2) An inventor spell belongs to a user whose user_id appears in the USPTO-LinkedIn crosswalk.
(3) Crosswalk user identifiers are deduplicated because a user can map to multiple inventors.
(4) The after-restrictions sample keeps spells with is_occupation_eligible equal to True.
(5) Inventor-spell shares use all employment spells in the corresponding sample as denominators.


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
    / "ResultsB04_InventorsInCandidateNewHires.csv"
)
AUDIT_COLUMNS = [
    "user_id",
    "id_user",
    "is_occupation_eligible",
]
INVENTOR_LINK_COLUMNS = [
    "user_id",
    "pv_inventor_id",
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
for column in INVENTOR_LINK_COLUMNS:
    inventor_links[column] = inventor_links[column].str.strip()
    flag_missing_identifier = inventor_links[column].isna() | inventor_links[column].eq("")
    if flag_missing_identifier.any():
        raise ValueError(
            f"The inventor crosswalk has {int(flag_missing_identifier.sum()):,} missing "
            f"values in {column}."
        )

flag_invalid_user_id = ~inventor_links["user_id"].str.fullmatch(r"\d+")
if flag_invalid_user_id.any():
    raise ValueError(
        f"The inventor crosswalk has {int(flag_invalid_user_id.sum()):,} nonnumeric user IDs."
    )

# Membership must be based on users, not crosswalk rows, because users can have multiple links.
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
# <> Step 4. Calculate inventor-spell counts and shares
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


is_inventor_spell = standardized_user_ids.isin(inventor_user_ids)
is_eligible_spell = spell_audit["is_occupation_eligible"]

sample_masks = {
    "Before job title restrictions": pd.Series(True, index=spell_audit.index),
    "After job title restrictions": is_eligible_spell,
}
summary_rows = []
for sample, sample_mask in sample_masks.items():
    employment_spell_count = int(sample_mask.sum())
    inventor_spell_count = int((sample_mask & is_inventor_spell).sum())
    summary_rows.append(
        {
            "sample": sample,
            "employment_spell_count": employment_spell_count,
            "inventor_spell_count": inventor_spell_count,
            "inventor_spell_share": inventor_spell_count / employment_spell_count,
        }
    )

inventor_spell_summary = pd.DataFrame(summary_rows)
inventor_spell_summary["sample"] = inventor_spell_summary["sample"].astype("string")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Save and report the summary
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
temporary_output = OUTPUT_FILE.with_suffix(f"{OUTPUT_FILE.suffix}.incomplete")
inventor_spell_summary.to_csv(temporary_output, index=False)
temporary_output.replace(OUTPUT_FILE)

display_summary = inventor_spell_summary.copy()
display_summary["inventor_spell_share"] = display_summary["inventor_spell_share"].map(
    lambda share: f"{share:.1%}"
)
print(display_summary.to_string(index=False))
print(f"Unique inventor-linked users in the crosswalk: {len(inventor_user_ids):,}")
print(f"Saved {len(inventor_spell_summary):,} rows: {OUTPUT_FILE}")
