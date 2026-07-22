"""
Task:
    Count firms with LinkedIn or Indeed postings in 2021-2022 and plot them by country.

Inputs:
    data/b_temp_data/A03_Industry_Automation/Automation_Companies.parquet
    data/b_temp_data/A03_Industry_Automation/Automation_LinkedInPostings.parquet
    data/b_temp_data/A03_Industry_Automation/Automation_IndeedPostings.parquet

Outputs:
    outputs/A03_Industry_Automation/B03_FirmPostingCounts_2021_2022.csv
    outputs/A03_Industry_Automation/B03_PostingDateDiagnostics.csv
    outputs/A03_Industry_Automation/B03_CompaniesWithPostingsByCountry.csv
    outputs/A03_Industry_Automation/B03_CompaniesWithPostingsByCountry.png

Notes:
(1) Posting counts combine LinkedIn and Indeed records dated from 2021 through 2022.
(2) Distinct job IDs are counted within source before the two source counts are added.
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

DATA_DIRECTORY = main.DIR_TEMPDATA / "A03_Industry_Automation"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A03_Industry_Automation"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_COMPANIES = DATA_DIRECTORY / "Automation_Companies.parquet"
POSTING_PATHS = {
    "LinkedIn": DATA_DIRECTORY / "Automation_LinkedInPostings.parquet",
    "Indeed": DATA_DIRECTORY / "Automation_IndeedPostings.parquet",
}
PATH_FIRM_COUNTS = OUTPUT_DIRECTORY / "B03_FirmPostingCounts_2021_2022.csv"
PATH_DATE_DIAGNOSTICS = OUTPUT_DIRECTORY / "B03_PostingDateDiagnostics.csv"
PATH_COUNTRY_COUNTS = OUTPUT_DIRECTORY / "B03_CompaniesWithPostingsByCountry.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "B03_CompaniesWithPostingsByCountry.png"

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2022-12-31")

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
# <> Step 2. Retain distinct 2021-2022 job postings within each source
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Source is part of the posting key because job IDs may overlap across providers.
(2) Missing job IDs cannot be verified as distinct and are excluded from posting counts.
(3) Date diagnostics are accumulated before the analysis-period restriction.
"""
posting_key_batches = []
date_diagnostic_records = []

for source, posting_path in POSTING_PATHS.items():
    for postings_batch in iterate_parquet_batches(
        posting_path,
        columns=["rcid", "job_id", "post_date"],
    ):
        postings_batch["post_date"], diagnostic = convert_date_with_diagnostics(
            postings_batch["post_date"],
            variable="post_date",
            dataset=f"Automation_{source}Postings",
        )
        date_diagnostic_records.append(diagnostic)

        in_period = postings_batch["post_date"].between(START_DATE, END_DATE)
        posting_keys = postings_batch.loc[in_period, ["rcid", "job_id"]].dropna()
        posting_keys["source"] = source
        posting_key_batches.append(posting_keys.drop_duplicates())

posting_keys = pd.concat(posting_key_batches, ignore_index=True)
posting_keys = posting_keys.drop_duplicates(["source", "job_id"], keep="first")


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
# <> Step 3. Count combined postings within firms
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


observed_firm_counts = posting_keys.groupby("rcid", as_index=False, observed=True).agg(
    posting_count=("job_id", "size")
)
firm_counts = companies.merge(observed_firm_counts, on="rcid", how="left", validate="one_to_one")
firm_counts["posting_count"] = firm_counts["posting_count"].fillna(0).astype("int64")
firm_counts.to_csv(PATH_FIRM_COUNTS, index=False)

positive_firms = firm_counts.loc[firm_counts["posting_count"].gt(0)].copy()
number_of_positive_firms = len(positive_firms)
print(f"Companies with positive combined 2021-2022 postings: {number_of_positive_firms:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Count and plot posting firms by headquarters country
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


country_counts = (
    positive_firms.groupby("hq_country", as_index=False, observed=True)
    .agg(company_count=("rcid", "nunique"))
    .sort_values(["company_count", "hq_country"], ascending=[False, True])
    .reset_index(drop=True)
)

if country_counts["company_count"].sum() != number_of_positive_firms:
    raise ValueError("Country counts do not sum to the number of firms with postings.")

country_counts.to_csv(PATH_COUNTRY_COUNTS, index=False)
save_country_bar(
    country_counts,
    country_column="hq_country",
    count_column="company_count",
    title=(
        "Number of automation companies across countries\n"
        "(Only companies with at least 1 observed job posting in 2021-22 included)"
    ),
    path=PATH_FIGURE,
)

print(country_counts.head(30).to_string(index=False))
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
