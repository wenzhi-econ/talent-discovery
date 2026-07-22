"""
Task:
    Count firms with observed employees in December 2022 and plot them by country.

Inputs:
    data/b_temp_data/A02_Industry_BioPharm/BioPharm_Companies.parquet
    data/b_temp_data/A02_Industry_BioPharm/BioPharm_UserPositions_FocalSpells.parquet
    codes/A02_Industry_BioPharm/A00_SummaryUtilities.py

Outputs:
    outputs/A02_Industry_BioPharm/B02_FirmEmployeeCounts_2022_12.csv
    outputs/A02_Industry_BioPharm/B02_EmployeeDateDiagnostics.csv
    outputs/A02_Industry_BioPharm/B02_CompaniesWithEmployeesByCountry.csv
    outputs/A02_Industry_BioPharm/B02_CompaniesWithEmployeesByCountry.png

Notes:
(1) The snapshot date is 2022-12-01 because the position dates have monthly precision.
(2) An employee is a distinct user with an active employment spell at the firm.
(3) Missing headquarters country is retained as a separate category.

Time: 2026-07-20
"""

from pathlib import Path
import sys
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths and parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

DATA_DIRECTORY = main.DIR_TEMPDATA / "A02_Industry_BioPharm"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A02_Industry_BioPharm"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_COMPANIES = DATA_DIRECTORY / "BioPharm_Companies.parquet"
PATH_POSITIONS = DATA_DIRECTORY / "BioPharm_UserPositions_FocalSpells.parquet"
PATH_FIRM_COUNTS = OUTPUT_DIRECTORY / "B02_FirmEmployeeCounts_2022_12.csv"
PATH_DATE_DIAGNOSTICS = OUTPUT_DIRECTORY / "B02_EmployeeDateDiagnostics.csv"
PATH_COUNTRY_COUNTS = OUTPUT_DIRECTORY / "B02_CompaniesWithEmployeesByCountry.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "B02_CompaniesWithEmployeesByCountry.png"

SNAPSHOT_DATE = pd.Timestamp("2022-12-01")


MISSING_COUNTRY = "Missing"
BATCH_SIZE = 500_000
FIGURE_DPI = 300
X_STEP = 1_000


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


def save_country_bar(
    counts: pd.DataFrame,
    country_column: str,
    count_column: str,
    title: str,
    path: Path,
    number_to_plot: int = 30,
) -> None:
    """Save a horizontal bar plot for the largest country groups."""
    plot_data = counts.head(number_to_plot).sort_values(count_column, ascending=True)
    figure, axis = plt.subplots(figsize=(10, 9))
    axis.barh(plot_data[country_column], plot_data[count_column], color="#4472C4")
    axis.set_xlabel("Number of companies")
    axis.set_ylabel("Country of headquarters")
    axis.set_title(title, loc="left")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    max_count = plot_data[count_column].max()
    axis.set_xticks(range(0, int(max_count) + 1, X_STEP))
    axis.set_xticks(range(int(X_STEP / 2), int(max_count) + 1, X_STEP), minor=True)
    axis.tick_params(axis="x", which="minor", length=3)
    axis.tick_params(axis="x", which="major", length=6)
    figure.tight_layout()
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
# <> Step 2. Identify active users in December 2022
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) The Parquet file is processed in batches to control memory use.
(2) A spell is active if it starts by the snapshot and has not ended before the snapshot.
(3) Date diagnostics count nonmissing strings that become missing during conversion.
(4) Duplicate user-firm pairs are removed again after batches are appended.
"""
active_user_firm_batches = []
date_diagnostic_records = []

for positions_batch in iterate_parquet_batches(
    PATH_POSITIONS,
    columns=["rcid", "user_id", "startdate", "enddate"],
):
    enddate_nonmissing = (positions_batch["enddate"].notna()) & (
        positions_batch["enddate"].astype("string").str.strip().ne("")
    )
    positions_batch["startdate"], start_diagnostic = convert_date_with_diagnostics(
        positions_batch["startdate"],
        variable="startdate",
        dataset="BioPharm_UserPositions_FocalSpells",
    )
    positions_batch["enddate"], end_diagnostic = convert_date_with_diagnostics(
        positions_batch["enddate"],
        variable="enddate",
        dataset="BioPharm_UserPositions_FocalSpells",
    )
    date_diagnostic_records.extend([start_diagnostic, end_diagnostic])

    valid_enddate_or_missing = ~enddate_nonmissing | positions_batch["enddate"].notna()
    active_spell = (
        positions_batch["startdate"].le(SNAPSHOT_DATE)
        & valid_enddate_or_missing
        & (~enddate_nonmissing | positions_batch["enddate"].ge(SNAPSHOT_DATE))
    )
    active_user_firm_batches.append(
        positions_batch.loc[active_spell, ["rcid", "user_id"]].drop_duplicates()
    )

active_user_firms = pd.concat(active_user_firm_batches, ignore_index=True).drop_duplicates()


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Combine and report date-conversion diagnostics
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
# <> Step 3. Count employees within firms and merge zero-employee firms
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


observed_firm_counts = active_user_firms.groupby("rcid", as_index=False, observed=True).agg(
    employee_count=("user_id", "nunique")
)
firm_counts = companies.merge(observed_firm_counts, on="rcid", how="left", validate="one_to_one")
firm_counts["employee_count"] = firm_counts["employee_count"].fillna(0).astype("int64")
firm_counts.to_csv(PATH_FIRM_COUNTS, index=False)

positive_firms = firm_counts.loc[firm_counts["employee_count"].gt(0)].copy()
number_of_positive_firms = len(positive_firms)
print(f"Companies with positive December 2022 employee counts: {number_of_positive_firms:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Count and plot employing firms by headquarters country
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


country_counts = (
    positive_firms.groupby("hq_country", as_index=False, observed=True)
    .agg(company_count=("rcid", "nunique"))
    .sort_values(["company_count", "hq_country"], ascending=[False, True])
    .reset_index(drop=True)
)

if country_counts["company_count"].sum() != number_of_positive_firms:
    raise ValueError("Country counts do not sum to the number of firms with employees.")

country_counts.to_csv(PATH_COUNTRY_COUNTS, index=False)
save_country_bar(
    country_counts,
    country_column="hq_country",
    count_column="company_count",
    title="Number of biotechnology and pharmaceutical companies across countries\n(Only companies with at least 1 observed employee as of December 2022 included)",
    path=PATH_FIGURE,
)

print(country_counts.head(30).to_string(index=False))
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
