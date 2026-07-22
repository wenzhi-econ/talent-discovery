"""
Task:
    Plot monthly LinkedIn postings for Scientist and Engineer jobs at Electronics firms.

Inputs:
    data/b_temp_data/A04_Industry_Electronics/Electronics_LinkedInPostings.parquet

Outputs:
    outputs/A04_Industry_Electronics/E02_LinkedInFocalPostingsMonthly.csv
    outputs/A04_Industry_Electronics/E02_LinkedInFocalPostingsDateDiagnostics.csv
    outputs/A04_Industry_Electronics/E02_LinkedInFocalPostingsMonthly.png

Notes:
(1) Focal postings have job category Scientist or Engineer.
(2) The monthly figure starts in January 2018.
(3) Posting counts use distinct nonmissing job IDs.
(4) The x- and y-axis ranges match the other source and overall-posting figures.
(5) Months without observed posting records remain missing and appear as gaps in the figure.

Time: 2026-07-21
"""

from pathlib import Path
import sys
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths and focal categories
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

FIGURE_DPI = 300
BATCH_SIZE = 500_000
FIGURE_START_MONTH = pd.Timestamp("2018-01-01")
FIGURE_END_MONTH = pd.Timestamp("2025-05-01")
Y_AXIS_MIN = 0
Y_AXIS_MAX = 1_000_000
Y_TICK_STEP = 200_000

DATA_DIRECTORY = main.DIR_TEMPDATA / "A04_Industry_Electronics"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A04_Industry_Electronics"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_POSTINGS = DATA_DIRECTORY / "Electronics_LinkedInPostings.parquet"
PATH_MONTHLY_COUNTS = OUTPUT_DIRECTORY / "E02_LinkedInFocalPostingsMonthly.csv"
PATH_DATE_DIAGNOSTICS = OUTPUT_DIRECTORY / "E02_LinkedInFocalPostingsDateDiagnostics.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "E02_LinkedInFocalPostingsMonthly.png"

FOCAL_JOB_CATEGORIES = ("Scientist", "Engineer")


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


def save_monthly_line(
    data: pd.DataFrame,
    title: str,
    path: Path,
    group_column: str | None = None,
) -> None:
    """Save an overall or country-specific monthly posting time series."""
    figure, axis = plt.subplots(figsize=(12, 6))

    if group_column is None:
        axis.plot(data["month"], data["posting_count"], color="#4472C4", linewidth=1.5)
    else:
        for group, group_data in data.groupby(group_column, sort=False, observed=True):
            axis.plot(
                group_data["month"],
                group_data["posting_count"],
                linewidth=1.4,
                label=group,
            )
        axis.legend(title="Posting country", frameon=False)

    axis.set_xlabel("Posting month")
    axis.set_ylabel("Number of job postings")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_xlim(FIGURE_START_MONTH, FIGURE_END_MONTH)
    axis.set_ylim(Y_AXIS_MIN, Y_AXIS_MAX)
    axis.set_yticks(range(Y_AXIS_MIN, Y_AXIS_MAX + 1, Y_TICK_STEP))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Read and retain distinct focal postings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


posting_batches = []
diagnostic_records = []

for postings_batch in iterate_parquet_batches(
    PATH_POSTINGS,
    columns=["job_id", "post_date", "job_category"],
):
    postings_batch["post_date"], diagnostic = convert_date_with_diagnostics(
        postings_batch["post_date"],
        variable="post_date",
        dataset="Electronics_LinkedInPostings",
    )
    diagnostic_records.append(diagnostic)

    focal_category = postings_batch["job_category"].isin(FOCAL_JOB_CATEGORIES)
    posting_batches.append(
        postings_batch.loc[focal_category, ["job_id", "post_date"]]
        .dropna()
        .drop_duplicates()
    )

postings = pd.concat(posting_batches, ignore_index=True)
postings = postings.drop_duplicates("job_id", keep="first")


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-1. Combine and report date-conversion diagnostics
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


date_diagnostics = pd.DataFrame(diagnostic_records)
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
# <> Step 2. Count focal postings by month and plot the time series
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


postings["month"] = postings["post_date"].dt.to_period("M").dt.to_timestamp()
monthly_counts = (
    postings.groupby("month", as_index=False, observed=True)
    .agg(posting_count=("job_id", "nunique"))
    .sort_values("month")
    .reset_index(drop=True)
)
monthly_counts.to_csv(PATH_MONTHLY_COUNTS, index=False)

"""
Notes:
(1) The CSV retains all available focal-posting months from the LinkedIn extract.
(2) The figure calendar runs from January 2018 through May 2025.
(3) Calendar months absent from the extract remain missing so that the plotted line breaks.
"""
figure_monthly_counts = pd.DataFrame(
    {"month": pd.date_range(FIGURE_START_MONTH, FIGURE_END_MONTH, freq="MS")}
).merge(
    monthly_counts,
    on="month",
    how="left",
    validate="one_to_one",
)

save_monthly_line(
    figure_monthly_counts,
    title=(
        "Monthly LinkedIn Scientist and Engineer postings by electronics companies"
    ),
    path=PATH_FIGURE,
)

print(monthly_counts.to_string(index=False))
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
