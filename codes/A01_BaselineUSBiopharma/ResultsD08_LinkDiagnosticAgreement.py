"""
Task:
    Plot link-diagnostic agreement under three nested link criteria.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageD_HirePostingLinks.parquet
    <== Constructed by
        "codes/A01_BaselineUSBiopharma/StageD_MatchedNewHiresAndJobPostings.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsD08_LinkDiagnosticAgreement.png

Descriptions of outputs:
(1) Output (a) contains six vertical bars: the shares of eligible links with equal state
    and equal normalized job title under each of three matching criteria.

Run:
    From the project root in the Talent environment:
    python -m codes.A01_BaselineUSBiopharma.ResultsD08_LinkDiagnosticAgreement

Notes:
(1) Missing comparisons remain in the eligible-link denominator and count as not equal.
(2) Agreement is mechanically 100 percent when the matching criterion imposes equality
    for the diagnostic being shown.
(3) Seniority is excluded because it is unavailable in the candidate-posting data.


Wang Wenzhi
Time: 2026-09-22
"""

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from codes import main as project
from codes.A01_BaselineUSBiopharma import ResultsD_Utils as plot


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define input, output, criteria, and diagnostic labels
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_LINKS = plot.STAGE_D_DIRECTORY / "StageD_HirePostingLinks.parquet"
OUTPUT_FILE = plot.OUTPUT_DIRECTORY / "ResultsD08_LinkDiagnosticAgreement.png"
CRITERIA = [
    "Only time",
    "Time + state",
    "Time + state + job title",
]
CRITERION_COLORS = [
    plot.COLOR_MATCHED,
    plot.COLOR_ACCENT,
    plot.COLOR_POSTINGS,
]
DIAGNOSTIC_LABELS = [
    "State",
    "Normalized job title",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Calculate agreement shares within each eligible-link sample
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
connection = duckdb.connect()
connection.execute("SET preserve_insertion_order = false")
diagnostic_summary = connection.execute(
    """
    SELECT
        1 AS criterion_order,
        'Only time' AS criterion,
        count(*) AS eligible_links,
        avg(CASE WHEN is_state_equal THEN 1.0 ELSE 0.0 END) AS state_share,
        avg(CASE WHEN is_title_equal THEN 1.0 ELSE 0.0 END) AS title_share
    FROM read_parquet(?)

    UNION ALL

    SELECT
        2 AS criterion_order,
        'Time + state' AS criterion,
        count(*) AS eligible_links,
        avg(CASE WHEN is_state_equal THEN 1.0 ELSE 0.0 END) AS state_share,
        avg(CASE WHEN is_title_equal THEN 1.0 ELSE 0.0 END) AS title_share
    FROM read_parquet(?)
    WHERE is_state_equal

    UNION ALL

    SELECT
        3 AS criterion_order,
        'Time + state + job title' AS criterion,
        count(*) AS eligible_links,
        avg(CASE WHEN is_state_equal THEN 1.0 ELSE 0.0 END) AS state_share,
        avg(CASE WHEN is_title_equal THEN 1.0 ELSE 0.0 END) AS title_share
    FROM read_parquet(?)
    WHERE is_state_equal AND is_title_equal

    ORDER BY criterion_order
    """,
    [str(INPUT_LINKS)] * 3,
).fetchdf()
connection.close()

required_columns = [
    "criterion_order",
    "criterion",
    "eligible_links",
    "state_share",
    "title_share",
]
plot.require_columns(diagnostic_summary, required_columns, name="Link diagnostic summary")
if not diagnostic_summary["criterion"].tolist() == CRITERIA:
    raise AssertionError("The link criteria are missing or out of order.")
if not diagnostic_summary["eligible_links"].gt(0).all():
    raise ValueError("Every link criterion must retain at least one eligible link.")
if (
    not diagnostic_summary[["state_share", "title_share"]]
    .map(lambda value: 0 <= value <= 1)
    .all(axis=None)
):
    raise AssertionError("Diagnostic agreement shares must lie between zero and one.")

plot_summary = diagnostic_summary.melt(
    id_vars=["criterion", "eligible_links"],
    value_vars=["state_share", "title_share"],
    var_name="diagnostic",
    value_name="agreement_share",
)
plot_summary["diagnostic"] = plot_summary["diagnostic"].map({
    "state_share": "State",
    "title_share": "Normalized job title",
})


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Plot and save the six agreement bars
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


figure, axis = plt.subplots(
    figsize=(plot.FIGURE_WIDTH_INCHES, plot.FIGURE_HEIGHT_INCHES),
    dpi=plot.FIGURE_DPI,
)
diagnostic_positions = np.arange(len(DIAGNOSTIC_LABELS), dtype="float64")
bar_width = 0.24
for criterion_index, (criterion, color) in enumerate(zip(CRITERIA, CRITERION_COLORS, strict=True)):
    criterion_rows = (
        plot_summary
        .loc[plot_summary["criterion"].eq(criterion)]
        .set_index("diagnostic")
        .loc[DIAGNOSTIC_LABELS]
    )
    eligible_links = int(criterion_rows["eligible_links"].iloc[0])
    bar_positions = diagnostic_positions + (criterion_index - 1) * bar_width
    bars = axis.bar(
        bar_positions,
        criterion_rows["agreement_share"],
        width=bar_width,
        color=color,
        label=f"{criterion} (N = {eligible_links:,})",
    )
    for bar, agreement_share in zip(
        bars,
        criterion_rows["agreement_share"],
        strict=True,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{agreement_share:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

axis.set_xticks(diagnostic_positions, DIAGNOSTIC_LABELS)
axis.set_ylim(0, 1.14)
axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
axis.set_ylabel("Share of eligible links with the same value", fontsize=13)
axis.set_title("Agreement in link diagnostics", fontsize=20, pad=16)
axis.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.09),
    frameon=False,
    fontsize=10.5,
    ncol=3,
)
plot.style_axis(axis, grid_axis="y")
figure.text(
    0.01,
    0.015,
    (
        "Notes: Missing comparisons count as not equal. Equality is imposed by the "
        "time + state and time + state + job title criteria as their names indicate."
    ),
    fontsize=9.5,
    color=plot.COLOR_TEXT_LIGHT,
)
figure.tight_layout(rect=(0, 0.13, 1, 1))
plot.save_figure(figure, OUTPUT_FILE)
plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Report the saved diagnostic figure
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


project.finish_run(
    run,
    "Compared state and title agreement under three nested link criteria.",
    f"Output: {project.relative_path(OUTPUT_FILE)}",
)
