"""
Task:
    Plot the December 2022 focal-employee distribution across employing Automation firms.

Inputs:
    data/b_temp_data/A03_Industry_Automation/Automation_Companies.parquet
    data/b_temp_data/A03_Industry_Automation/Automation_UserPositions_FocalSpells.parquet

Outputs:
    outputs/A03_Industry_Automation/C03_FirmFocalEmployeeCounts_2022_12.csv
    outputs/A03_Industry_Automation/C03_FocalEmployeeDateDiagnostics.csv
    outputs/A03_Industry_Automation/C03_FocalEmployeeDistributionAllCompanies.png

Notes:
(1) Focal employees have job category Scientist or Engineer.
(2) A focal employee is a distinct active user at the firm on 2022-12-01.
(3) Companies above the 97.5th percentile of positive focal counts are excluded.
(4) Histogram bars have width one employee, and both axes use linear scales.
(5) Figure notes report statistics from all companies with positive focal counts.

Wang Wenzhi, with the help of CODEX
Time: 2026-07-21
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths and parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

DATA_DIRECTORY = main.DIR_TEMPDATA / "A03_Industry_Automation"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A03_Industry_Automation"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_COMPANIES = DATA_DIRECTORY / "Automation_Companies.parquet"
PATH_POSITIONS = DATA_DIRECTORY / "Automation_UserPositions_FocalSpells.parquet"
PATH_FIRM_COUNTS = OUTPUT_DIRECTORY / "C03_FirmFocalEmployeeCounts_2022_12.csv"
PATH_DATE_DIAGNOSTICS = OUTPUT_DIRECTORY / "C03_FocalEmployeeDateDiagnostics.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "C03_FocalEmployeeDistributionAllCompanies.png"

SNAPSHOT_DATE = pd.Timestamp("2022-12-01")
FOCAL_JOB_CATEGORIES = ("Scientist", "Engineer")

MISSING_COUNTRY = "Missing"
BATCH_SIZE = 500_000
FIGURE_DPI = 300
TRUNCATION_QUANTILE = 0.975
X_STEP = 10


def iterate_parquet_batches(path: Path, columns: list[str]):
    """Yield selected Parquet columns as pandas DataFrames of manageable size."""
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(columns=columns, batch_size=BATCH_SIZE):
        yield record_batch.to_pandas()


def convert_date_with_diagnostics(
    values: pd.Series,
    variable: str,
    dataset: str,
) -> tuple[pd.Series, dict[str, object]]:
    """Convert one string date column and describe nonmissing conversion failures."""
    nonmissing = values.notna() & values.astype("string").str.strip().ne("")
    converted = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    failed = nonmissing & converted.isna()
    nonmissing_count = int(nonmissing.sum())
    failed_count = int(failed.sum())
    failed_share = failed_count / nonmissing_count if nonmissing_count else 0.0

    diagnostic = {
        "dataset": dataset,
        "variable": variable,
        "row_count": int(len(values)),
        "missing_or_blank_count": int((~nonmissing).sum()),
        "nonmissing_string_count": nonmissing_count,
        "failed_conversion_count": failed_count,
        "failed_conversion_share": failed_share,
    }
    return converted, diagnostic


def print_date_diagnostics(diagnostics: pd.DataFrame) -> None:
    """Print date diagnostics with percentages that are easy to inspect."""
    display = diagnostics.copy()
    display["failed_conversion_percent"] = 100 * display["failed_conversion_share"]
    columns = [
        "dataset",
        "variable",
        "nonmissing_string_count",
        "failed_conversion_count",
        "failed_conversion_percent",
    ]
    print("Date conversion diagnostics:")
    print(display[columns].to_string(index=False))


def save_integer_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    figure_note: str,
    path: Path,
) -> None:
    """Save a width-one histogram with linear x- and y-axes."""
    integer_values = values.astype("int64")
    frequencies = integer_values.value_counts(sort=False).sort_index()

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        frequencies.index,
        frequencies.values,
        width=1,
        align="center",
        color="#4472C4",
        edgecolor="none",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Number of companies")
    axis.set_title(title, loc="left")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    max_count = integer_values.max()
    axis.set_xlim(0, max_count + 1)
    axis.set_xticks(range(0, int(max_count) + 1, X_STEP))
    axis.set_xticks(range(int(X_STEP / 2), int(max_count) + 1, X_STEP), minor=True)
    axis.tick_params(axis="x", which="minor", length=3)
    axis.tick_params(axis="x", which="major", length=6)
    figure.tight_layout(rect=(0, 0.14, 1, 1))
    axes_position = axis.get_position()
    figure.text(
        axes_position.x0,
        0.03,
        figure_note,
        ha="left",
        va="bottom",
        fontsize=8,
    )
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read the company universe
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


companies = pd.read_parquet(PATH_COMPANIES, columns=["rcid", "hq_country"])
if companies["rcid"].isna().any() or companies["rcid"].duplicated().any():
    raise ValueError("Company rcid must be nonmissing and unique.")

companies["hq_country"] = companies["hq_country"].astype("string").str.strip()
companies["hq_country"] = companies["hq_country"].mask(
    companies["hq_country"].eq(""),
    MISSING_COUNTRY,
)
companies["hq_country"] = companies["hq_country"].fillna(MISSING_COUNTRY)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Identify focal employees active in December 2022
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Load data in batches and keep only occupations of interest
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


"""
Notes:
(1) Rows are first restricted to focal job categories to reduce later memory use.
(2) The snapshot restriction follows the same inclusive rules as B02.
(3) Duplicate user-firm pairs are removed after all Parquet batches are appended.
"""
focal_user_firm_batches = []
date_diagnostic_records = []

for positions_batch in iterate_parquet_batches(
    PATH_POSITIONS,
    columns=["rcid", "user_id", "startdate", "enddate", "job_category"],
):
    enddate_nonmissing = positions_batch["enddate"].notna() & (
        positions_batch["enddate"].astype("string").str.strip().ne("")
    )
    positions_batch["startdate"], start_diagnostic = convert_date_with_diagnostics(
        positions_batch["startdate"],
        variable="startdate",
        dataset="Automation_UserPositions_FocalSpells",
    )
    positions_batch["enddate"], end_diagnostic = convert_date_with_diagnostics(
        positions_batch["enddate"],
        variable="enddate",
        dataset="Automation_UserPositions_FocalSpells",
    )
    date_diagnostic_records.extend([start_diagnostic, end_diagnostic])

    focal_category = positions_batch["job_category"].isin(FOCAL_JOB_CATEGORIES)
    valid_enddate_or_missing = ~enddate_nonmissing | positions_batch["enddate"].notna()
    active_spell = (
        positions_batch["startdate"].le(SNAPSHOT_DATE)
        & valid_enddate_or_missing
        & (~enddate_nonmissing | positions_batch["enddate"].ge(SNAPSHOT_DATE))
    )
    focal_user_firm_batches.append(
        positions_batch.loc[
            focal_category & active_spell,
            ["rcid", "user_id"],
        ].drop_duplicates()
    )

focal_user_firms = pd.concat(focal_user_firm_batches, ignore_index=True).drop_duplicates()


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Combine and report date-conversion diagnostics
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


date_diagnostics = pd.DataFrame(date_diagnostic_records)
date_diagnostics = date_diagnostics.groupby(
    ["dataset", "variable"], as_index=False, observed=True
).agg(
    row_count=("row_count", "sum"),
    missing_or_blank_count=("missing_or_blank_count", "sum"),
    nonmissing_string_count=("nonmissing_string_count", "sum"),
    failed_conversion_count=("failed_conversion_count", "sum"),
)
date_diagnostics["failed_conversion_share"] = (
    date_diagnostics["failed_conversion_count"]
    / date_diagnostics["nonmissing_string_count"].replace(0, pd.NA)
).fillna(0.0)
date_diagnostics.to_csv(PATH_DATE_DIAGNOSTICS, index=False)
print_date_diagnostics(date_diagnostics)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Count focal employees within firms and restore zero-count firms
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


observed_counts = focal_user_firms.groupby("rcid", as_index=False, observed=True).agg(
    focal_employee_count=("user_id", "nunique")
)
firm_counts = companies.merge(observed_counts, on="rcid", how="left", validate="one_to_one")
firm_counts["focal_employee_count"] = firm_counts["focal_employee_count"].fillna(0).astype("int64")
firm_counts.to_csv(PATH_FIRM_COUNTS, index=False)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Restrict the figure sample and determine the truncation value
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Companies without an observed focal employee are excluded from the figure.
(2) The upper cutoff is the 97.5th percentile among companies with positive focal counts.
(3) ``interpolation="higher"`` chooses an observed integer count as the cutoff.
(4) Summary statistics are calculated before the upper-tail truncation is applied.
"""
positive_firms = firm_counts.loc[firm_counts["focal_employee_count"].ge(1)].copy()
positive_focal_counts = positive_firms["focal_employee_count"]

full_sample_mean = positive_focal_counts.mean()
full_sample_median = positive_focal_counts.median()
full_sample_p25 = positive_focal_counts.quantile(0.25)
full_sample_p50 = positive_focal_counts.quantile(0.50)
full_sample_p75 = positive_focal_counts.quantile(0.75)
full_sample_standard_deviation = positive_focal_counts.std()

truncation_value = int(
    positive_focal_counts.quantile(
        TRUNCATION_QUANTILE,
        interpolation="higher",
    )
)
figure_firms = positive_firms.loc[
    positive_firms["focal_employee_count"].le(truncation_value)
].copy()
number_truncated = len(positive_firms) - len(figure_firms)

print(f"Companies with at least 1 observed focal employee: {len(positive_firms):,}")
print(f"97.5th-percentile truncation value: {truncation_value:,} focal employees")
print(f"Companies above the truncation value and excluded: {number_truncated:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Plot the focal-employee distribution
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


save_integer_histogram(
    figure_firms["focal_employee_count"],
    title=(
        "The distribution of observed scientists and engineers in December 2022 across "
        "automation companies"
    ),
    xlabel="Number of observed scientist and engineer employees at the company",
    figure_note=(
        "Notes:\n"
        f"(1) The plot includes companies with 1-{truncation_value:,} focal employees; "
        f"{number_truncated:,} companies above the 97.5th-percentile cutoff are truncated.\n"
        f"(2) Statistics with all {len(positive_firms):,} companies with positive scientist and engineer counts: "
        f"mean = {full_sample_mean:,.2f}; "
        f"median = {full_sample_median:,.0f}; p25 = {full_sample_p25:,.0f}; "
        f"p50 = {full_sample_p50:,.0f}; p75 = {full_sample_p75:,.0f}; "
        f"standard deviation = {full_sample_standard_deviation:,.2f}."
    ),
    path=PATH_FIGURE,
)

print(positive_focal_counts.describe().to_string())
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
