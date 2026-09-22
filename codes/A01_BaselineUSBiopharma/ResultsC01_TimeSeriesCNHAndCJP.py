"""
Task:
    Plot monthly candidate new hires and candidate LinkedIn job postings over time.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_CandidateNewHires.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
(b) data/b_temp_data/A01_BaselineUSBiopharma/StageC_CandidateLinkedInPostings.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageC3_MergeTwoPartsTogether.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsC01_TimeSeriesCNHAndCJP.png

Descriptions of outputs:
(1) Output (a) plots monthly candidate-new-hire counts by start month and monthly candidate
    LinkedIn posting counts by publication month.

Run:
    From the project root in the Talent environment:
    python -m codes.A01_BaselineUSBiopharma.ResultsC01_TimeSeriesCNHAndCJP

Notes:
(1) Each series is plotted only over the month range covered by its input dataset.
(2) A month with no records inside a dataset's covered range receives a count of zero.
(3) Posting counts reflect company-specific Stage C extraction windows, not all market postings.
(4) The deterministic PNG output is replaced after the temporary image is validated.


Wang Wenzhi
Time: 2026-09-22
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from PIL import Image

from codes import main as project


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths, columns, and figure settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


STAGE_DIRECTORY = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
INPUT_HIRES = STAGE_DIRECTORY / "StageB_CandidateNewHires.parquet"
INPUT_POSTINGS = STAGE_DIRECTORY / "StageC_CandidateLinkedInPostings.parquet"
OUTPUT_FILE = (
    project.DIR_OUTPUTS
    / "A01_BaselineUSBiopharma"
    / "ResultsC01_TimeSeriesCNHAndCJP.png"
)
HIRE_COLUMNS = [
    "id_user",
    "id_rcid",
    "start_date",
]
POSTING_COLUMNS = [
    "id_job",
    "publication_date",
]
HIRE_KEY = [
    "id_user",
    "id_rcid",
]
FIGURE_WIDTH_INCHES = 12.8
FIGURE_HEIGHT_INCHES = 7.2
FIGURE_DPI = 200
COLOR_HIRES = "#003399"
COLOR_POSTINGS = "#CC5500"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read and validate the two candidate samples
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


run = project.start_run(__file__)
candidate_hires = pd.read_parquet(INPUT_HIRES, columns=HIRE_COLUMNS)
candidate_postings = pd.read_parquet(INPUT_POSTINGS, columns=POSTING_COLUMNS)

if candidate_hires.empty or candidate_postings.empty:
    raise ValueError("The candidate-new-hire and candidate-posting inputs must be nonempty.")
if not candidate_hires.columns.is_unique or not candidate_postings.columns.is_unique:
    raise ValueError("The input datasets must have unique column names.")
if candidate_hires[HIRE_KEY].isna().any(axis=None):
    raise ValueError("Candidate new hires contain missing user or company identifiers.")
if candidate_hires.duplicated(HIRE_KEY).any():
    raise ValueError("Candidate new hires contain duplicate user-company observations.")
if candidate_postings["id_job"].isna().any():
    raise ValueError("Candidate LinkedIn postings contain missing posting identifiers.")
if candidate_postings["id_job"].duplicated().any():
    raise ValueError("Candidate LinkedIn postings contain duplicate posting identifiers.")

candidate_hires["start_date"] = pd.to_datetime(
    candidate_hires["start_date"],
    errors="raise",
)
candidate_postings["publication_date"] = pd.to_datetime(
    candidate_postings["publication_date"],
    errors="raise",
)
if candidate_hires["start_date"].isna().any():
    raise ValueError("Candidate new hires contain missing start dates.")
if candidate_postings["publication_date"].isna().any():
    raise ValueError("Candidate LinkedIn postings contain missing publication dates.")
if candidate_hires["start_date"].ne(candidate_hires["start_date"].dt.normalize()).any():
    raise ValueError("Candidate-new-hire start dates must be normalized calendar dates.")
if candidate_postings["publication_date"].ne(
    candidate_postings["publication_date"].dt.normalize()
).any():
    raise ValueError("Candidate-posting publication dates must be normalized calendar dates.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Construct complete monthly count series
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


candidate_hires["month"] = candidate_hires["start_date"].dt.to_period("M").dt.to_timestamp()
candidate_postings["month"] = (
    candidate_postings["publication_date"].dt.to_period("M").dt.to_timestamp()
)

hire_month_range = pd.date_range(
    start=candidate_hires["month"].min(),
    end=candidate_hires["month"].max(),
    freq="MS",
)
posting_month_range = pd.date_range(
    start=candidate_postings["month"].min(),
    end=candidate_postings["month"].max(),
    freq="MS",
)
monthly_hire_counts = (
    candidate_hires.groupby("month").size().reindex(hire_month_range, fill_value=0)
)
monthly_posting_counts = (
    candidate_postings.groupby("month").size().reindex(posting_month_range, fill_value=0)
)
monthly_hire_counts.index.name = "month"
monthly_hire_counts.name = "candidate_new_hire_count"
monthly_posting_counts.index.name = "month"
monthly_posting_counts.name = "candidate_linkedin_posting_count"

if int(monthly_hire_counts.sum()) != len(candidate_hires):
    raise AssertionError("Monthly candidate-new-hire counts do not sum to the input row count.")
if int(monthly_posting_counts.sum()) != len(candidate_postings):
    raise AssertionError("Monthly candidate-posting counts do not sum to the input row count.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Draw and save the monthly time series
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


figure, axis = plt.subplots(
    figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
    dpi=FIGURE_DPI,
)
axis.plot(
    monthly_hire_counts.index,
    monthly_hire_counts,
    color=COLOR_HIRES,
    linewidth=2.4,
    marker="o",
    markersize=4,
    label="Candidate new hires (start month)",
)
axis.plot(
    monthly_posting_counts.index,
    monthly_posting_counts,
    color=COLOR_POSTINGS,
    linewidth=2.4,
    marker="s",
    markersize=3.8,
    label="Candidate LinkedIn postings (posting month)",
)

first_month = min(monthly_hire_counts.index.min(), monthly_posting_counts.index.min())
last_month = max(monthly_hire_counts.index.max(), monthly_posting_counts.index.max())
axis.set_xlim(first_month, last_month)
axis.set_ylim(bottom=0)
axis.set_title(
    "Candidate new hires and LinkedIn job postings over time",
    fontsize=20,
    pad=16,
)
axis.set_ylabel("Number of observations per month", fontsize=13)
axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axis.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
axis.tick_params(axis="both", labelsize=11)
axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
axis.grid(axis="x", visible=False)
axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.legend(loc="upper left", frameon=False, fontsize=12)

figure.text(
    0.01,
    0.015,
    (
        "Notes: Each line follows its own sample period. Posting counts reflect "
        "company-specific Stage C extraction windows."
    ),
    fontsize=9.5,
    color="#4D4D4D",
)
figure.tight_layout(rect=(0, 0.055, 1, 1))

project.ensure_parent(OUTPUT_FILE)
temporary_output = OUTPUT_FILE.with_suffix(f"{OUTPUT_FILE.suffix}.incomplete")
figure.savefig(temporary_output, format="png", dpi=FIGURE_DPI, facecolor="white")
plt.close(figure)

with Image.open(temporary_output) as saved_image:
    saved_image.verify()
temporary_output.replace(OUTPUT_FILE)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Report the sample coverage and saved figure
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


project.finish_run(
    run,
    f"Candidate new hires: {len(candidate_hires):,}.",
    f"Candidate LinkedIn postings: {len(candidate_postings):,}.",
    f"Output: {project.relative_path(OUTPUT_FILE)}",
)
