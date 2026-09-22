"""
Task:
    Plot Stage D match coverage under three nested link criteria.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageD_FullCandidateNewHires.parquet
(b) data/b_temp_data/A01_BaselineUSBiopharma/StageD_FullCandidateJobPostings.parquet
(c) data/b_temp_data/A01_BaselineUSBiopharma/StageD_HirePostingLinks.parquet
    <== Constructed by
        "codes/A01_BaselineUSBiopharma/StageD_MatchedNewHiresAndJobPostings.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsD01_MatchedSampleCoverage.png

Descriptions of outputs:
(1) Output (a) contains nine vertical bars: coverage of candidate new hires, candidate
    job postings, and companies with candidate new hires under each of three criteria.

Run:
    From the project root in the Talent environment:
    python -m codes.A01_BaselineUSBiopharma.ResultsD01_MatchedSampleCoverage

Notes:
(1) The criteria are time only; time plus equal state; and time plus equal state and
    normalized job title.
(2) A company is matched when at least one candidate new hire has an eligible posting.
(3) Missing diagnostic comparisons do not satisfy an equality restriction.


Wang Wenzhi
Time: 2026-09-22
"""

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.ticker import PercentFormatter

from codes import main as project
from codes.A01_BaselineUSBiopharma import ResultsD_Utils as plot


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define inputs, output, and matching criteria
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_HIRES = plot.STAGE_D_DIRECTORY / "StageD_FullCandidateNewHires.parquet"
INPUT_POSTINGS = plot.STAGE_D_DIRECTORY / "StageD_FullCandidateJobPostings.parquet"
INPUT_LINKS = plot.STAGE_D_DIRECTORY / "StageD_HirePostingLinks.parquet"
OUTPUT_FILE = plot.OUTPUT_DIRECTORY / "ResultsD01_MatchedSampleCoverage.png"
CRITERIA = [
    (1, "Only time"),
    (2, "Time + state"),
    (3, "Time + state + job title"),
]
CRITERION_COLORS = [
    plot.COLOR_MATCHED,
    plot.COLOR_ACCENT,
    plot.COLOR_POSTINGS,
]
SAMPLE_LABELS = [
    "Candidate\nnew hires",
    "Candidate\njob postings",
    "Companies with\ncandidate new hires",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read candidate-sample denominators
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
candidate_hire_companies = pd.read_parquet(INPUT_HIRES, columns=["id_rcid"])
plot.require_columns(
    candidate_hire_companies,
    ["id_rcid"],
    name="Candidate-new-hire sample",
)
if candidate_hire_companies["id_rcid"].isna().any():
    raise ValueError("The candidate-new-hire sample contains missing company identifiers.")

candidate_counts = {
    "Candidate\nnew hires": pq.ParquetFile(INPUT_HIRES).metadata.num_rows,
    "Candidate\njob postings": pq.ParquetFile(INPUT_POSTINGS).metadata.num_rows,
    "Companies with\ncandidate new hires": candidate_hire_companies["id_rcid"].nunique(),
}
if any(count <= 0 for count in candidate_counts.values()):
    raise ValueError("All candidate-sample denominators must be positive.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Derive endpoint match levels from the link diagnostics
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


connection = duckdb.connect()
connection.execute("SET preserve_insertion_order = false")
connection.read_parquet(str(INPUT_LINKS)).select(
    "id_position, id_job, id_rcid, is_state_equal, is_title_equal"
).create_view("eligible_links")
connection.execute(
    """
    CREATE TEMP TABLE hire_match_levels AS
    SELECT
        id_position,
        id_rcid,
        max(
            CASE
                WHEN is_state_equal AND is_title_equal THEN 3
                WHEN is_state_equal THEN 2
                ELSE 1
            END
        ) AS match_level
    FROM eligible_links
    GROUP BY id_position, id_rcid
    """
)
connection.execute(
    """
    CREATE TEMP TABLE posting_match_levels AS
    SELECT
        id_job,
        max(
            CASE
                WHEN is_state_equal AND is_title_equal THEN 3
                WHEN is_state_equal THEN 2
                ELSE 1
            END
        ) AS match_level
    FROM eligible_links
    GROUP BY id_job
    """
)
connection.execute(
    """
    CREATE TEMP TABLE company_match_levels AS
    SELECT id_rcid, max(match_level) AS match_level
    FROM hire_match_levels
    GROUP BY id_rcid
    """
)

matched_counts = {}
for sample, table_name in [
    ("Candidate\nnew hires", "hire_match_levels"),
    ("Candidate\njob postings", "posting_match_levels"),
    ("Companies with\ncandidate new hires", "company_match_levels"),
]:
    counts = connection.execute(
        f"""
        SELECT
            count(*) FILTER (WHERE match_level >= 1),
            count(*) FILTER (WHERE match_level >= 2),
            count(*) FILTER (WHERE match_level >= 3)
        FROM {table_name}
        """
    ).fetchone()
    matched_counts[sample] = [int(value) for value in counts]
connection.close()

coverage_rows = []
for sample in SAMPLE_LABELS:
    denominator = candidate_counts[sample]
    for criterion_index, (_, criterion) in enumerate(CRITERIA):
        matched_count = matched_counts[sample][criterion_index]
        coverage_rows.append({
            "sample": sample,
            "criterion": criterion,
            "candidate_count": denominator,
            "matched_count": matched_count,
            "matched_share": matched_count / denominator,
        })
coverage_summary = pd.DataFrame(coverage_rows)
if not coverage_summary["matched_share"].between(0, 1).all():
    raise AssertionError("Calculated match shares must lie between zero and one.")
for sample in SAMPLE_LABELS:
    sample_counts = coverage_summary.loc[
        coverage_summary["sample"].eq(sample),
        "matched_count",
    ].to_numpy()
    if not np.all(sample_counts[:-1] >= sample_counts[1:]):
        raise AssertionError(f"Coverage is not nested for {sample.replace(chr(10), ' ')}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Plot and save the nine coverage bars
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


figure, axis = plt.subplots(
    figsize=(plot.FIGURE_WIDTH_INCHES, plot.FIGURE_HEIGHT_INCHES),
    dpi=plot.FIGURE_DPI,
)
sample_positions = np.arange(len(SAMPLE_LABELS), dtype="float64")
bar_width = 0.24
for criterion_index, ((_, criterion), color) in enumerate(
    zip(CRITERIA, CRITERION_COLORS, strict=True)
):
    criterion_rows = (
        coverage_summary
        .loc[coverage_summary["criterion"].eq(criterion)]
        .set_index("sample")
        .loc[SAMPLE_LABELS]
    )
    bar_positions = sample_positions + (criterion_index - 1) * bar_width
    bars = axis.bar(
        bar_positions,
        criterion_rows["matched_share"],
        width=bar_width,
        color=color,
        label=criterion,
    )
    for bar, row in zip(bars, criterion_rows.itertuples(), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{row.matched_share:.1%}\n({row.matched_count:,})",
            ha="center",
            va="bottom",
            fontsize=9.3,
        )

axis.set_xticks(sample_positions, SAMPLE_LABELS)
axis.set_ylim(0, 1.14)
axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
axis.set_ylabel("Share covered by the matched sample", fontsize=13)
axis.set_title("Coverage of the Stage D matched samples", fontsize=20, pad=16)
axis.legend(loc="upper right", frameon=False, fontsize=10.5)
plot.style_axis(axis, grid_axis="y")
figure.text(
    0.01,
    0.015,
    (
        "Notes: Labels report coverage rates and matched counts. Missing state or title "
        "comparisons do not satisfy the corresponding equality criterion."
    ),
    fontsize=9.5,
    color=plot.COLOR_TEXT_LIGHT,
)
figure.tight_layout(rect=(0, 0.055, 1, 1))
plot.save_figure(figure, OUTPUT_FILE)
plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Report saved coverage results
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


project.finish_run(
    run,
    "Compared coverage under three nested link criteria.",
    f"Output: {project.relative_path(OUTPUT_FILE)}",
)
