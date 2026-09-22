"""
Task:
    Create a word cloud of standardized job titles in the candidate-new-hire sample.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_CandidateNewHires.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsB03_CandidateNewHireJobTitles.png

Descriptions of outputs:
(1) Output (a) is a word cloud of standardized titles among candidate new hires, with one input
    observation per user-company pair.

Run:
    Run from the project root after activating the Talent environment:
    python codes/A01_BaselineUSBiopharma/ResultsB03_WordCloudsInCandidateNewHires.py

Notes:
(1) Each complete standardized job title is one word-cloud item, even when it contains several
    words.
(2) Repeated titles receive proportionally more weight because the figure describes candidate new
    hires rather than distinct title strings.
(3) Titles are not split into individual words, and no stopwords are removed from within a title.
(4) The random seed and plotting settings match the Stage B employment-spell word clouds.
(5) The deterministic PNG output is replaced on every run after the temporary file is validated.


Wang Wenzhi
Time: 2026-09-21
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from wordcloud import WordCloud


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths and figure settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_HIRES = (
    PROJECT_ROOT
    / "data"
    / "b_temp_data"
    / "A01_BaselineUSBiopharma"
    / "StageB_CandidateNewHires.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "A01_BaselineUSBiopharma"
OUTPUT_FILE = OUTPUT_DIR / "ResultsB03_CandidateNewHireJobTitles.png"
REQUIRED_COLUMNS = [
    "id_user",
    "id_rcid",
    "job_title_normalized",
    "is_occupation_eligible",
]
PAIR_COLUMNS = [
    "id_user",
    "id_rcid",
]
FIGURE_WIDTH_INCHES = 12.8
FIGURE_HEIGHT_INCHES = 7.2
FIGURE_DPI = 200
WORD_CLOUD_WIDTH = 2_400
WORD_CLOUD_HEIGHT = 1_240
RANDOM_SEED = 20260921


def save_word_cloud(job_titles: pd.Series, *, output_file: Path) -> None:
    """
    Create, validate, and save the candidate-new-hire word-cloud figure.

    Parameters
    ----------
    job_titles : pd.Series
        Non-missing standardized titles, with one element per user-company observation.
    output_file : Path
        Final PNG path.

    Raises
    ------
    ValueError
        Raised when no usable title text is available.
    """
    title_frequencies = job_titles.astype("string").value_counts().sort_index()
    if title_frequencies.empty:
        raise ValueError("No usable candidate-new-hire job titles are available.")

    cloud = WordCloud(
        width=WORD_CLOUD_WIDTH,
        height=WORD_CLOUD_HEIGHT,
        background_color="white",
        colormap="viridis",
        max_words=500,
        prefer_horizontal=0.9,
        random_state=RANDOM_SEED,
    ).generate_from_frequencies(title_frequencies.to_dict())

    figure, axis = plt.subplots(
        figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES),
        dpi=FIGURE_DPI,
    )
    axis.imshow(cloud, interpolation="bilinear")
    axis.set_axis_off()
    axis.set_title("Candidate new hires: job titles", fontsize=24, pad=12)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)

    temporary_output = output_file.with_suffix(f"{output_file.suffix}.incomplete")
    figure.savefig(temporary_output, format="png", dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)

    with Image.open(temporary_output) as saved_image:
        saved_image.verify()
    temporary_output.replace(output_file)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read and validate the candidate-new-hire sample
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


candidate_hires = pd.read_parquet(INPUT_HIRES, columns=REQUIRED_COLUMNS)

if not candidate_hires.columns.is_unique:
    raise ValueError("The candidate-new-hire file has duplicate column names.")
if candidate_hires[PAIR_COLUMNS].isna().any(axis=None):
    raise ValueError(
        "The candidate-new-hire file has missing user or company identifiers."
    )
if candidate_hires.duplicated(PAIR_COLUMNS).any():
    raise ValueError(
        "The candidate-new-hire file has duplicate user-company observations."
    )
if candidate_hires["is_occupation_eligible"].isna().any():
    raise ValueError(
        "The candidate-new-hire eligibility indicator contains missing values."
    )
if not pd.api.types.is_bool_dtype(candidate_hires["is_occupation_eligible"]):
    raise TypeError(
        "The candidate-new-hire eligibility indicator must have Boolean dtype."
    )
if not candidate_hires["is_occupation_eligible"].all():
    raise ValueError(
        "The candidate-new-hire file contains title-ineligible observations."
    )
if candidate_hires["job_title_normalized"].isna().any():
    raise ValueError(
        "The candidate-new-hire file contains missing standardized job titles."
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Create the candidate-new-hire word cloud
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_word_cloud(
    candidate_hires["job_title_normalized"],
    output_file=OUTPUT_FILE,
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Report the saved figure
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


print(f"Candidate new hires with non-missing titles: {len(candidate_hires):,}")
print(
    "Distinct standardized titles: "
    f"{candidate_hires['job_title_normalized'].nunique(dropna=True):,}"
)
print(f"Saved candidate-new-hire word cloud: {OUTPUT_FILE}")
