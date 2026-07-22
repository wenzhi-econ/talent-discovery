"""
Task:
    Count Electronics companies and plot the 30 largest headquarters-country groups.

Inputs:
    data/b_temp_data/A04_Industry_Electronics/Electronics_Companies.parquet

Outputs:
    outputs/A04_Industry_Electronics/B01_CompanyCountsByCountry.csv
    outputs/A04_Industry_Electronics/B01_CompanyCountsByCountry.png

Run:
    conda activate Talent
    python codes/A04_Industry_Electronics/B01_CompanyCountsByCountry.py

Notes:
(1) A company is identified by its Revelio ``rcid``.
(2) Missing headquarters country is retained as a separate category.
(3) The CSV includes every country, while the figure displays the largest 30 groups.

Wang Wenzhi, with the help of CODEX
Time: 2026-07-20
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths and parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

DATA_DIRECTORY = main.DIR_TEMPDATA / "A04_Industry_Electronics"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A04_Industry_Electronics"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_COMPANIES = DATA_DIRECTORY / "Electronics_Companies.parquet"
PATH_COUNTS = OUTPUT_DIRECTORY / "B01_CompanyCountsByCountry.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "B01_CompanyCountsByCountry.png"

FIGURE_DPI = 300
MISSING_COUNTRY = "Missing"
X_STEP = 2_000


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
# <> Step 1. Read and validate the company universe
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Only the company identifier and headquarters country are needed for this summary.
(2) The checks ensure that rows and companies have a one-to-one relationship.
"""
companies = pd.read_parquet(PATH_COMPANIES, columns=["rcid", "hq_country"])

if companies["rcid"].isna().any():
    raise ValueError("The company extract contains missing rcid values.")
if companies["rcid"].duplicated().any():
    raise ValueError("The company extract contains duplicate rcid values.")

total_companies = companies["rcid"].nunique()
print(f"Total number of Electronics companies: {total_companies:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Count and plot companies by headquarters country
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Blank country strings are treated as missing along with null values.
(2) Alphabetical country names break ties in company counts reproducibly.
"""
companies["hq_country"] = companies["hq_country"].astype("string").str.strip()
companies["hq_country"] = companies["hq_country"].mask(
    companies["hq_country"].eq(""),
    MISSING_COUNTRY,
)
companies["hq_country"] = companies["hq_country"].fillna(MISSING_COUNTRY)

country_counts = (
    companies.groupby("hq_country", as_index=False, observed=True)
    .agg(company_count=("rcid", "nunique"))
    .sort_values(["company_count", "hq_country"], ascending=[False, True])
    .reset_index(drop=True)
)

if country_counts["company_count"].sum() != total_companies:
    raise ValueError("Country counts do not sum to the total number of companies.")

country_counts.to_csv(PATH_COUNTS, index=False)
save_country_bar(
    country_counts,
    country_column="hq_country",
    count_column="company_count",
    title=(
        "Number of electronics companies across countries\n"
        "(All companies -- active or non-active -- included)"
    ),
    path=PATH_FIGURE,
)

print(country_counts.head(30).to_string(index=False))
print(f"Saved table: {main.relative_path(PATH_COUNTS)}")
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
