"""
Task:
    Plot the December 2022 employee-count distribution across employing BioPharm firms.

Inputs:
    outputs/A02_Industry_BioPharm/B02_FirmEmployeeCounts_2022_12.csv

Outputs:
    outputs/A02_Industry_BioPharm/C01_EmployeeDistributionAllCompanies.png

Notes:
(1) The distribution includes firms with at least one observed employee.
(2) Companies above the 97.5th percentile of positive employee counts are excluded.
(3) Histogram bars have width one employee, and both axes use linear scales.
(4) Figure notes report statistics from all companies with positive employee counts.

Wang Wenzhi, with the help of CODEX
Time: 2026-07-21
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A02_Industry_BioPharm"
PATH_FIRM_COUNTS = OUTPUT_DIRECTORY / "B02_FirmEmployeeCounts_2022_12.csv"
PATH_FIGURE = OUTPUT_DIRECTORY / "C01_EmployeeDistributionAllCompanies.png"

FIGURE_DPI = 300
TRUNCATION_QUANTILE = 0.975
X_STEP = 10


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
# <> Step 1. Read and validate firm employee counts
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


firm_counts = pd.read_csv(PATH_FIRM_COUNTS, usecols=["rcid", "employee_count"])
if firm_counts["rcid"].duplicated().any() or firm_counts["employee_count"].lt(0).any():
    raise ValueError("Firm employee counts must be unique by rcid and nonnegative.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Restrict the figure sample and determine the truncation value
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Companies without an observed employee are excluded as requested.
(2) The upper cutoff is the 97.5th percentile among companies with positive counts.
(3) ``interpolation="higher"`` chooses an observed integer count as the cutoff.
(4) Summary statistics are calculated before the upper-tail truncation is applied.
"""
positive_firms = firm_counts.loc[firm_counts["employee_count"].ge(1)].copy()
positive_employee_counts = positive_firms["employee_count"]

full_sample_mean = positive_employee_counts.mean()
full_sample_median = positive_employee_counts.median()
full_sample_p25 = positive_employee_counts.quantile(0.25)
full_sample_p50 = positive_employee_counts.quantile(0.50)
full_sample_p75 = positive_employee_counts.quantile(0.75)
full_sample_standard_deviation = positive_employee_counts.std()

truncation_value = int(
    positive_employee_counts.quantile(
        TRUNCATION_QUANTILE,
        interpolation="higher",
    )
)
figure_firms = positive_firms.loc[positive_firms["employee_count"].le(truncation_value)].copy()
number_truncated = len(positive_firms) - len(figure_firms)

print(f"Companies with at least 1 observed employee: {len(positive_firms):,}")
print(f"97.5th-percentile truncation value: {truncation_value:,} employees")
print(f"Companies above the truncation value and excluded: {number_truncated:,}")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Plot the employee-count distribution
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


save_integer_histogram(
    figure_firms["employee_count"],
    title=(
        "The distribution of observed firm size in December 2022 across biotechnology and "
        "pharmaceutical companies"
    ),
    xlabel="Number of observed employees at the company",
    figure_note=(
        "Notes:\n"
        f"(1) The plot includes companies with 1-{truncation_value:,} observed employees; "
        f"{number_truncated:,} companies above the 97.5th-percentile cutoff are truncated.\n"
        f"(2) Statistics with all {len(positive_firms):,} companies with positive employee counts: "
        f"mean = {full_sample_mean:,.2f}; "
        f"median = {full_sample_median:,.0f}; p25 = {full_sample_p25:,.0f}; "
        f"p50 = {full_sample_p50:,.0f}; p75 = {full_sample_p75:,.0f}; "
        f"standard deviation = {full_sample_standard_deviation:,.2f}."
    ),
    path=PATH_FIGURE,
)

print(positive_employee_counts.describe().to_string())
print(f"Saved figure: {main.relative_path(PATH_FIGURE)}")
