"""
Task:
    Create word clouds for eligible and non-eligible Stage B employment-spell job titles.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"

Outputs:
(a) outputs/A01_BaselineUSBiopharma/ResultsB02_EligibleJobTitles.png
(b) outputs/A01_BaselineUSBiopharma/ResultsB02_NonEligibleJobTitles.png

Descriptions of outputs:
(1) Output (a) is a word cloud of standardized titles among eligible employment spells.
(2) Output (b) is a word cloud of standardized titles among non-eligible employment spells.

Run:
    Run from the project root after activating the Talent environment:
    python codes/A01_BaselineUSBiopharma/ResultsB02_WordCloudsInJobTitleRestrictions.py

Notes:
(1) Each complete standardized job title is one word-cloud item, even when it contains several
    words.
(2) Repeated titles receive proportionally more weight because the figures describe spells rather
    than distinct title strings.
(3) Titles are not split into individual words, and no stopwords are removed from within a title.
(4) The random seed and plotting settings are fixed so that repeated runs reproduce the figures.
(5) The deterministic PNG outputs are replaced on every run after temporary files are validated.


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
INPUT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "b_temp_data"
    / "A01_BaselineUSBiopharma"
    / "StageB_SpellAudit.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "A01_BaselineUSBiopharma"
OUTPUT_ELIGIBLE = OUTPUT_DIR / "ResultsB02_EligibleJobTitles.png"
OUTPUT_NON_ELIGIBLE = OUTPUT_DIR / "ResultsB02_NonEligibleJobTitles.png"
REQUIRED_COLUMNS = [
    "job_title_normalized",
    "is_occupation_eligible",
]
FIGURE_WIDTH_INCHES = 12.8
FIGURE_HEIGHT_INCHES = 7.2
FIGURE_DPI = 200
WORD_CLOUD_WIDTH = 2_400
WORD_CLOUD_HEIGHT = 1_240
RANDOM_SEED = 20260921


def save_word_cloud(
    job_titles: pd.Series,
    *,
    output_file: Path,
    figure_title: str,
    colormap: str,
) -> None:
    """
    Create, validate, and save one deterministic word-cloud figure.

    Parameters
    ----------
    job_titles : pd.Series
        Non-missing standardized titles, with one element per employment spell.
    output_file : Path
        Final PNG path.
    figure_title : str
        Title shown above the word cloud.
    colormap : str
        Matplotlib colormap used by the word cloud.

    Raises
    ------
    ValueError
        Raised when no usable title text is available.
    """
    title_frequencies = job_titles.dropna().astype("string").value_counts().sort_index()
    if title_frequencies.empty:
        raise ValueError(f"No usable job titles are available for {figure_title}.")

    cloud = WordCloud(
        width=WORD_CLOUD_WIDTH,
        height=WORD_CLOUD_HEIGHT,
        background_color="white",
        colormap=colormap,
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
    axis.set_title(figure_title, fontsize=24, pad=12)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)

    temporary_output = output_file.with_suffix(f"{output_file.suffix}.incomplete")
    figure.savefig(temporary_output, format="png", dpi=FIGURE_DPI, facecolor="white")
    plt.close(figure)

    with Image.open(temporary_output) as saved_image:
        saved_image.verify()
    temporary_output.replace(output_file)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read and validate the Stage B spell audit
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spell_audit = pd.read_parquet(INPUT_AUDIT, columns=REQUIRED_COLUMNS)

if not spell_audit.columns.is_unique:
    raise ValueError("The Stage B spell audit has duplicate column names.")
if spell_audit["is_occupation_eligible"].isna().any():
    raise ValueError("The Stage B eligibility indicator contains missing values.")
if not pd.api.types.is_bool_dtype(spell_audit["is_occupation_eligible"]):
    raise TypeError("The Stage B eligibility indicator must have Boolean dtype.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Create the eligible and non-eligible word clouds
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
eligible_titles = spell_audit.loc[
    spell_audit["is_occupation_eligible"],
    "job_title_normalized",
]
non_eligible_titles = spell_audit.loc[
    ~spell_audit["is_occupation_eligible"],
    "job_title_normalized",
]

save_word_cloud(
    eligible_titles,
    output_file=OUTPUT_ELIGIBLE,
    figure_title="Eligible employment spells",
    colormap="viridis",
)
save_word_cloud(
    non_eligible_titles,
    output_file=OUTPUT_NON_ELIGIBLE,
    figure_title="Non-eligible employment spells",
    colormap="OrRd",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Report the saved figures
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


print(f"Eligible spells with non-missing titles: {eligible_titles.notna().sum():,}")
print(f"Saved eligible-title word cloud: {OUTPUT_ELIGIBLE}")
print(
    f"Non-eligible spells with non-missing titles: {non_eligible_titles.notna().sum():,}"
)
print(f"Saved non-eligible-title word cloud: {OUTPUT_NON_ELIGIBLE}")
