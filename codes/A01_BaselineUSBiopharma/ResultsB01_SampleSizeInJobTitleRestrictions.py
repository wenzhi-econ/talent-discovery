"""
Task:
    Summarize employment spells and distinct standardized job titles before and after the Stage B
    job-title restrictions.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsB01_SampleSizeInJobTitleRestrictions.csv

Descriptions of outputs:
(1) Output (a) has one row for the sample before the job-title restrictions and one row for the
    eligible sample after the restrictions.

Run:
    Run from the project root after activating the Talent environment:
    python codes/A01_BaselineUSBiopharma/ResultsB01_SampleSizeInJobTitleRestrictions.py

Notes:
(1) A distinct standardized job title is a distinct non-missing value of job_title_normalized.
(2) The before-restrictions sample contains all US Biopharma spells in the Stage B spell audit.
(3) The after-restrictions sample keeps spells with is_occupation_eligible equal to True.
(4) The deterministic CSV output is replaced on every run after the temporary file is written.


Wang Wenzhi
Time: 2026-09-21
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
OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "A01_BaselineUSBiopharma"
    / "ResultsB01_SampleSizeInJobTitleRestrictions.csv"
)
REQUIRED_COLUMNS = [
    "job_title_normalized",
    "is_occupation_eligible",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read and validate the Stage B spell audit
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spell_audit = pd.read_parquet(INPUT_AUDIT, columns=REQUIRED_COLUMNS)

if not spell_audit.columns.is_unique:
    raise ValueError("The Stage B spell audit has duplicate column names.")
if spell_audit["is_occupation_eligible"].isna().any():
    raise ValueError("The Stage B eligibility indicator contains missing values.")
if not pd.api.types.is_bool_dtype(spell_audit["is_occupation_eligible"]):
    raise TypeError("The Stage B eligibility indicator must have Boolean dtype.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Count employment spells and distinct standardized job titles
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


eligible_spells = spell_audit.loc[spell_audit["is_occupation_eligible"]].copy()

sample_size_summary = pd.DataFrame(
    {
        "sample": pd.Series(
            [
                "Before job title restrictions",
                "After job title restrictions",
            ],
            dtype="string",
        ),
        "employment_spell_count": [
            len(spell_audit),
            len(eligible_spells),
        ],
        "standardized_job_title_count": [
            spell_audit["job_title_normalized"].nunique(dropna=True),
            eligible_spells["job_title_normalized"].nunique(dropna=True),
        ],
        "missing_standardized_job_title_count": [
            int(spell_audit["job_title_normalized"].isna().sum()),
            int(eligible_spells["job_title_normalized"].isna().sum()),
        ],
    }
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Save and report the summary
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
temporary_output = OUTPUT_FILE.with_suffix(f"{OUTPUT_FILE.suffix}.incomplete")
sample_size_summary.to_csv(temporary_output, index=False)
temporary_output.replace(OUTPUT_FILE)

print(sample_size_summary.to_string(index=False))
print(f"Saved {len(sample_size_summary):,} rows: {OUTPUT_FILE}")
