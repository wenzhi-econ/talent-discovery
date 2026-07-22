"""
Task:
    Identify active Electronics firms and plot the 30 largest country groups.

Inputs:
    outputs/A04_Industry_Electronics/B02_FirmEmployeeCounts_2022_12.csv
    outputs/A04_Industry_Electronics/B03_FirmPostingCounts_2021_2022.csv

Outputs:
    outputs/A04_Industry_Electronics/B04_ActiveFirmCounts.csv
    outputs/A04_Industry_Electronics/B04_ActiveCompaniesByCountry.csv
    outputs/A04_Industry_Electronics/B04_ActiveCompaniesByCountry.png

Notes:
(1) An active firm has positive December 2022 employees and positive combined postings.
(2) The posting measure combines LinkedIn and Indeed records from 2021 through 2022.
(3) Missing headquarters country is retained as a separate category.

Time: 2026-07-20
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A04_Industry_Electronics"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_EMPLOYEE_COUNTS = OUTPUT_DIRECTORY / "B02_FirmEmployeeCounts_2022_12.csv"
PATH_POSTING_COUNTS = OUTPUT_DIRECTORY / "B03_FirmPostingCounts_2021_2022.csv"
PATH_ACTIVE_COUNTS = OUTPUT_DIRECTORY / "B04_ActiveFirmCounts.csv"
PATH_COUNTRY_COUNTS = OUTPUT_DIRECTORY / "B04_ActiveCompaniesByCountry.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "B04_ActiveCompaniesByCountry.png"

FIGURE_DPI = 300
X_STEP = 1_000


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
# <> Step 1. Combine firm-level employee and posting counts
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) B02 and B03 must be run before this script.
(2) A one-to-one merge ensures that both inputs contain exactly one row per company.
"""
employee_counts = pd.read_csv(PATH_EMPLOYEE_COUNTS)
posting_counts = pd.read_csv(
    PATH_POSTING_COUNTS,
    usecols=["rcid", "posting_count"],
)

firm_counts = employee_counts.merge(
    posting_counts,
    on="rcid",
    how="inner",
    validate="one_to_one",
)
if len(firm_counts) != len(employee_counts):
    raise ValueError("Employee and posting count files do not contain the same company universe.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Define active firms
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


firm_counts["active_firm"] = firm_counts["employee_count"].gt(0) & firm_counts["posting_count"].gt(
    0
)
firm_counts.to_csv(PATH_ACTIVE_COUNTS, index=False)

active_firms = firm_counts.loc[firm_counts["active_firm"]].copy()
number_of_active_firms = len(active_firms)
print(f"Total number of active Electronics companies: {number_of_active_firms:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Count and plot active firms by headquarters country
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


country_counts = (
    active_firms.groupby("hq_country", as_index=False, observed=True)
    .agg(company_count=("rcid", "nunique"))
    .sort_values(["company_count", "hq_country"], ascending=[False, True])
    .reset_index(drop=True)
)

if country_counts["company_count"].sum() != number_of_active_firms:
    raise ValueError("Country counts do not sum to the total number of active firms.")

country_counts.to_csv(PATH_COUNTRY_COUNTS, index=False)
save_country_bar(
    country_counts,
    country_column="hq_country",
    count_column="company_count",
    title=(
        "Number of electronics companies across countries\n"
        "(Only active companies included)\n"
        "(Active companies: at least 1 observed employee as of December 2022 and 1 job "
        "posting in 2021-22)"
    ),
    path=PATH_FIGURE,
)

print(country_counts.head(30).to_string(index=False))
print(f"Saved outputs in: {main.relative_path(PATH_FIGURE)}")
