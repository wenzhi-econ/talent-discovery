"""
Task:
    Plot eligible prior postings per matched new hire under three link criteria.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageD_HirePostingLinks.parquet
    <== Constructed by
        "codes/A01_BaselineUSBiopharma/StageD_MatchedNewHiresAndJobPostings.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsD05_PostingsPerMatchedHire.png

Descriptions of outputs:
(1) Output (a) uses three panels to plot empirical cumulative distributions of eligible
    prior-posting counts among matched candidate new hires.

Run:
    From the project root in the Talent environment:
    python -m codes.A01_BaselineUSBiopharma.ResultsD05_PostingsPerMatchedHire

Notes:
(1) Each panel conditions on being matched under its own criterion and uses a linear axis.
(2) Each panel has its own horizontal range. Counts above its 99th percentile are grouped
    at the panel's display cap.
(3) The figure reports matched N and P25, P50, P75, P90, P95, and P99 for every criterion.
(4) Missing diagnostic comparisons do not satisfy an equality restriction.


Wang Wenzhi
Time: 2026-09-22
"""

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

from codes import main as project
from codes.A01_BaselineUSBiopharma import ResultsD_Utils as plot


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define input, output, criteria, and display settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_LINKS = plot.STAGE_D_DIRECTORY / "StageD_HirePostingLinks.parquet"
OUTPUT_FILE = plot.OUTPUT_DIRECTORY / "ResultsD05_PostingsPerMatchedHire.png"
CRITERIA = [
    ("Only time", "time_count", plot.COLOR_MATCHED),
    ("Time + state", "state_count", plot.COLOR_ACCENT),
    ("Time + state + job title", "state_title_count", plot.COLOR_POSTINGS),
]
PERCENTILES = [
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]
PERCENTILE_LABELS = [
    "P25",
    "P50",
    "P75",
    "P90",
    "P95",
    "P99",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Count eligible postings for each candidate new hire
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
connection = duckdb.connect()
connection.execute("SET preserve_insertion_order = false")
partner_counts = connection.execute(
    """
    SELECT
        id_position,
        CAST(count(*) AS BIGINT) AS time_count,
        CAST(count(*) FILTER (WHERE is_state_equal) AS BIGINT) AS state_count,
        CAST(
            count(*) FILTER (WHERE is_state_equal AND is_title_equal)
            AS BIGINT
        ) AS state_title_count
    FROM read_parquet(?)
    GROUP BY id_position
    """,
    [str(INPUT_LINKS)],
).fetchdf()
connection.close()

required_columns = ["id_position"] + [column for _, column, _ in CRITERIA]
plot.require_columns(partner_counts, required_columns, name="New-hire partner counts")
if partner_counts[required_columns].isna().any(axis=None):
    raise ValueError("New-hire partner counts contain missing values.")
if not (
    partner_counts["time_count"].ge(partner_counts["state_count"])
    & partner_counts["state_count"].ge(partner_counts["state_title_count"])
    & partner_counts["state_title_count"].ge(0)
).all():
    raise AssertionError("New-hire partner counts are not nested across criteria.")

matched_counts = {
    label: partner_counts.loc[partner_counts[column].gt(0), column].copy()
    for label, column, _ in CRITERIA
}
if any(counts.empty for counts in matched_counts.values()):
    raise ValueError("Every link criterion must match at least one candidate new hire.")
summary_statistics = {}
for label, counts in matched_counts.items():
    percentile_values = np.quantile(counts, PERCENTILES, method="higher").astype("int64")
    summary_statistics[label] = {
        "N": len(counts),
        **dict(zip(PERCENTILE_LABELS, percentile_values, strict=True)),
    }


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Plot and save three readable empirical cumulative-distribution panels
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


figure, axes = plt.subplots(
    nrows=1,
    ncols=len(CRITERIA),
    sharey=True,
    figsize=(plot.FIGURE_WIDTH_INCHES, plot.FIGURE_HEIGHT_INCHES),
    dpi=plot.FIGURE_DPI,
)
for panel_index, (axis, (label, _, color)) in enumerate(zip(axes, CRITERIA, strict=True)):
    counts = matched_counts[label]
    statistics = summary_statistics[label]
    display_cap = statistics["P99"]
    display_counts = counts.clip(upper=display_cap)
    sorted_counts, cumulative_probability = plot.empirical_cdf(display_counts)
    axis.axvspan(
        statistics["P25"],
        statistics["P75"],
        color=color,
        alpha=0.10,
        linewidth=0,
    )
    axis.axvline(
        statistics["P50"],
        color=color,
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
    )
    axis.axvline(
        statistics["P90"],
        color=color,
        linestyle=":",
        linewidth=1.7,
        alpha=0.85,
    )
    axis.step(
        sorted_counts,
        cumulative_probability,
        where="post",
        color=color,
        linewidth=2.4,
    )
    axis.set_xlim(0, max(display_cap, 1))
    axis.set_ylim(0, 1.01)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.set_title(f"{label}\nN = {len(counts):,}", fontsize=12.5, pad=10)
    plot.style_axis(axis, grid_axis="both")
    if panel_index == 0:
        axis.set_ylabel("Cumulative share of matched new hires", fontsize=12)

summary_header = f"{'Criterion':<27}{'Matched N':>12}" + "".join(
    f"{percentile:>9}" for percentile in PERCENTILE_LABELS
)
summary_rows = ["Matched-sample partner-count summary", summary_header]
for label, _, _ in CRITERIA:
    statistics = summary_statistics[label]
    summary_rows.append(
        f"{label:<27}{statistics['N']:>12,}"
        + "".join(f"{statistics[percentile]:>9,}" for percentile in PERCENTILE_LABELS)
    )

figure.suptitle("Matched job postings per matched new hire", fontsize=20, y=0.97)
figure.supxlabel("Number of matched job postings", fontsize=12.5, y=0.275)
figure.text(
    0.06,
    0.055,
    "\n".join(summary_rows),
    family="monospace",
    fontsize=8.7,
    linespacing=1.35,
    va="bottom",
)
figure.text(
    0.06,
    0.015,
    (
        "Notes: Each panel conditions on matching under its criterion and uses its own "
        "linear axis ending at P99; larger counts are grouped at P99.\n"
        "Shading marks P25-P75; dashed and dotted lines mark P50 and P90."
    ),
    fontsize=9,
    color=plot.COLOR_TEXT_LIGHT,
)
figure.subplots_adjust(left=0.07, right=0.985, bottom=0.34, top=0.82, wspace=0.12)
plot.save_figure(figure, OUTPUT_FILE)
plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Report the saved posting-count distributions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


project.finish_run(
    run,
    "Compared postings per matched hire under three nested link criteria.",
    f"Output: {project.relative_path(OUTPUT_FILE)}",
)
