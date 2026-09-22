"""
Task:
    Construct company posting-extraction windows from the full Stage B hire sample.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_CandidateNewHires.parquet
    <== Constructed by StageB_CandidateUSBiopharmaNewHires.py.

Outputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageC1_CompanyWindows.parquet

Descriptions of outputs:
(1) Output (a) is one row per company with inclusive posting bounds and selected-hire count.

Run:
conda run -s -n Talent python -m codes.A01_BaselineUSBiopharma.StageC1_CompanyThresholdsForPostingDates

Notes:
(1) Upload output (a) to `Files/WenzhiW/A01_BaselineUSBiopharma/StageC1_CompanyWindows.parquet` on
    Fabric.
(2) Subtract 12 calendar months from earliest hire and 1 calendar month from latest hire.
(3) Month subtraction clips to the last valid target day; both endpoints are included.
(4) Company windows deliberately include gaps; Stage D enforces the hire-specific window.
(5) Outputs are replaced only when OVERWRITE_OUTPUTS is explicitly enabled.


Wang Wenzhi
Time: 2026-09-08
"""

import time
from pathlib import Path

import pandas as pd

from codes import main as project

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define the input and upload destination
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_HIRES = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma" / "StageB_CandidateNewHires.parquet"
OUTPUT_WINDOWS = INPUT_HIRES.parent / "StageC1_CompanyWindows.parquet"
OVERWRITE_OUTPUTS = True
LINK_RULE_VERSION = "company_calendar_1_12_inclusive_v1"
PAIR_COLUMNS = ["id_user", "id_rcid"]
MISSING_TEXT = ("", "empty", "null", "none", "nan", "na", "n/a")


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Require the named columns and reject duplicate column labels."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing or not frame.columns.is_unique:
        raise ValueError(f"Missing columns: {missing}; columns must also be unique.")


def clean_text(values: pd.Series) -> pd.Series:
    """Trim text and replace explicit missing tokens with pandas missing values."""
    cleaned = values.astype("string").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin(MISSING_TEXT))


def standardize_id(values: pd.Series) -> pd.Series:
    """Canonicalize signed integral identifiers without passing through floats."""
    if pd.api.types.is_float_dtype(values) and values.dropna().abs().gt(2**53 - 1).any():
        raise ValueError("Unsafe floating identifier; retrieve an integer or string source.")
    cleaned = clean_text(values)
    valid = cleaned.str.fullmatch(r"-?\d+(?:\.0+)?", na=True)
    if not valid.all():
        bad_values = cleaned.loc[~valid].drop_duplicates().head(5).tolist()
        raise ValueError(f"Identifiers must be signed integers: {bad_values}")
    cleaned = cleaned.str.replace(r"\.0+$", "", regex=True)
    is_negative = cleaned.str.startswith("-").fillna(False)
    digits = cleaned.str.removeprefix("-").str.lstrip("0")
    digits = digits.mask(digits.eq("").fillna(False), "0")
    return digits.mask(is_negative & digits.ne("0").fillna(False), "-" + digits)


def require_unique(frame: pd.DataFrame, keys: list[str]) -> None:
    """Require nonmissing unique keys, including for a valid empty frame."""
    require_columns(frame, keys)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate keys: {keys}.")


def calendar_bounds(hire_dates: pd.Series) -> pd.DataFrame:
    """Return inclusive dates 12 and 1 calendar months before each hire date."""
    dates = pd.to_datetime(hire_dates, errors="raise")
    if dates.isna().any() or dates.ne(dates.dt.normalize()).any():
        raise ValueError("Calendar bounds require nonmissing normalized daily dates.")
    return pd.DataFrame(
        {
            "posting_lower": dates - pd.DateOffset(months=12),
            "posting_upper": dates - pd.DateOffset(months=1),
        },
        index=dates.index,
    )


def save_parquet(frame: pd.DataFrame, output_file: Path, *, overwrite: bool) -> None:
    """Validate a staged Parquet round trip before replacing the destination."""
    if output_file.exists() and not overwrite:
        raise FileExistsError(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    staged = output_file.with_suffix(".parquet.incomplete")
    normalized = frame.reset_index(drop=True)
    normalized.to_parquet(staged, index=False)
    restored = pd.read_parquet(staged)
    pd.testing.assert_frame_equal(normalized, restored, check_dtype=False)
    for attempt in range(5):
        try:
            staged.replace(output_file)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.25 * 2**attempt)
    print(f"Saved {len(normalized):,} rows: {output_file}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Obtain company-specific windows for job postings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def construct_company_windows(candidate_hires: pd.DataFrame) -> pd.DataFrame:
    """
    Construct inclusive company extraction windows from selected Stage B hires.

    Parameters
    ----------
    candidate_hires : pd.DataFrame
        One selected Stage B row per standardized user-company pair.

    Returns
    -------
    pd.DataFrame
        One row per standardized company with posting bounds and hire counts.
    """
    required_columns = [
        "id_user",
        "id_rcid",
        "rcid",
        "start_date",
        "is_occupation_eligible",
    ]
    require_columns(candidate_hires, required_columns)
    if candidate_hires.empty:
        raise ValueError("The Stage B candidate-hire input is empty.")

    hires = candidate_hires.copy()
    require_unique(hires, PAIR_COLUMNS)
    hires["id_rcid"] = standardize_id(hires["id_rcid"])
    standardized_raw_rcid = standardize_id(hires["rcid"])
    if not hires["id_rcid"].eq(standardized_raw_rcid).all():
        raise ValueError("Stage B id_rcid does not match the standardized source rcid.")
    if not hires["is_occupation_eligible"].fillna(False).all():
        raise ValueError("Occupation-ineligible rows reached the Stage B hire output.")

    hires["start_date"] = pd.to_datetime(hires["start_date"], errors="raise")
    hire_bounds = calendar_bounds(hires["start_date"])
    hires = hires.assign(**{column: hire_bounds[column] for column in hire_bounds})
    company_windows = (
        hires
        .groupby("id_rcid", as_index=False)
        .agg(
            rcid=("rcid", "first"),
            earliest_hire_date=("start_date", "min"),
            latest_hire_date=("start_date", "max"),
            posting_lower=("posting_lower", "min"),
            posting_upper=("posting_upper", "max"),
            hire_count=("id_user", "size"),
        )
        .sort_values("id_rcid")
        .reset_index(drop=True)
    )
    company_windows["link_rule_version"] = pd.Series(
        LINK_RULE_VERSION, index=company_windows.index, dtype="string"
    )

    require_unique(company_windows, ["id_rcid"])
    if company_windows["posting_lower"].gt(company_windows["posting_upper"]).any():
        raise AssertionError("An extraction window is reversed.")
    if int(company_windows["hire_count"].sum()) != len(hires):
        raise AssertionError("Company hire counts do not sum to the Stage B input count.")
    return company_windows


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Save the company-level posting windows
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def main() -> None:
    """Read Stage B, construct company windows, and save the validated local output."""
    run = project.start_run(__file__)
    if OUTPUT_WINDOWS.exists() and not OVERWRITE_OUTPUTS:
        raise FileExistsError(OUTPUT_WINDOWS)
    candidate_hires = pd.read_parquet(INPUT_HIRES)
    company_windows = construct_company_windows(candidate_hires)
    save_parquet(company_windows, OUTPUT_WINDOWS, overwrite=OVERWRITE_OUTPUTS)
    project.finish_run(run, f"Upload the {len(company_windows):,} company windows to Fabric.")


if __name__ == "__main__":
    main()
